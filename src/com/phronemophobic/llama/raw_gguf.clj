(ns com.phronemophobic.llama.raw-gguf
  (:require [com.phronemophobic.clong.gen.jna :as gen]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [com.phronemophobic.llama.impl.model :as model]
            [com.rpl.specter :as specter])
  (:import java.lang.ref.Cleaner
           com.sun.jna.Memory
           java.nio.charset.CodingErrorAction
           java.nio.charset.CharsetDecoder
           java.nio.charset.Charset
           java.nio.ByteBuffer
           java.nio.CharBuffer
           com.sun.jna.Memory
           com.sun.jna.Pointer
           com.sun.jna.ptr.IntByReference
           com.sun.jna.ptr.FloatByReference
           com.sun.jna.Structure)
  (:gen-class))

(def cleaner (Cleaner/create))

(defn ^:private write-edn [w obj]
  (binding [*print-length* nil
            *print-level* nil
            *print-dup* false
            *print-meta* false
            *print-readably* true

            ;; namespaced maps not part of edn spec
            *print-namespace-maps* false

            *out* w]
    (pr obj)))

(def libllama-options
  {com.sun.jna.Library/OPTION_STRING_ENCODING "UTF8"})
(def ^:no-doc libllama
  (try
    (com.sun.jna.NativeLibrary/getInstance "llama-gguf" libllama-options)
    (catch UnsatisfiedLinkError e
      ;; to support local builds
      (let [libllama (com.sun.jna.NativeLibrary/getInstance "llama" libllama-options)]
        ;; Make sure it's not an old version
        (try
          (.getFunction ^com.sun.jna.NativeLibrary libllama
                        "llama_token_to_piece")
          (catch UnsatisfiedLinkError _
            ;; throw original error
            (throw e)))
        libllama))))

(defn ^:private dump-api []
  (let [outf (io/file
              "resources"
              "com"
              "phronemophobic"
              "llama"
              "api-raw.edn")]
    (.mkdirs (.getParentFile outf))
    (with-open [w (io/writer outf)]
      (write-edn w
                 ((requiring-resolve 'com.phronemophobic.clong.clang/easy-api)
                  "/Users/adrian/workspace/llama.cpp/llama.h")
                 ))))


(def api
  #_((requiring-resolve 'com.phronemophobic.clong.clang/easy-api) "/Users/adrian/workspace/llama.cpp/llama.h")
  (with-open [rdr (io/reader
                     (io/resource
                      "com/phronemophobic/llama/api-gguf.edn"))
                rdr (java.io.PushbackReader. rdr)]
      (edn/read rdr)))

(gen/def-api libllama api)

(let [struct-prefix (gen/ns-struct-prefix *ns*)]
  (defmacro import-structs! []
    `(gen/import-structs! api ~struct-prefix)))

(import-structs!)



(defn ^:private ->bool [b]
  (if b
    (byte 1)
    (byte 0)))

(defn ^:private ->float-array-by-reference [v]
  (let [arr (float-array v)
        arrlen (alength arr)
        num-bytes (* arrlen 4)
        mem (doto (Memory. num-bytes)
              (.write 0 arr 0 arrlen))
        fbr (doto (FloatByReference.)
              (.setPointer mem))]
    fbr))

(defn ^:private map->llama-context-params [m]
  (reduce-kv
   (fn [^llama_context_params
        params k v]
     (case k
       :seed (.writeField params "seed" (int v))
       :n-ctx (.writeField params "n_ctx" (int v))
       :n-batch (.writeField params "n_batch" (int v))
       :n-threads (.writeField params "n_threads" (int v))
       :n-threads-batch (.writeField params "n_threads_batch" (int v))

       :rope-freq-base (.writeField params "rope_freq_base" (float v))
       :rope-freq-scale (.writeField params "rope_freq_scale" (float v))

       :mul_mat_q (.writeField params "mul_mat_q" (->bool v))
       :f16-kv (.writeField params "f16_kv" (->bool v))
       :logits-all (.writeField params "logits_all" (->bool v))
       :embedding (.writeField params "embedding" (->bool v))

       ;; ignore unknown keys
       nil)
     ;; return params
     params)
   (llama_context_default_params)
   m))

(defn ^:private map->llama-model-params [m]
  (reduce-kv
   (fn [^llama_model_params
        params k v]
     (case k
       :n-gpu-layers (.writeField params "n_gpu_layers" (int v))
       :main-gpu (.writeField params "main_gpu" (int v))
       :tensor-split (.writeField params "tensor_split" (->float-array-by-reference  v))

       :vocab-only (.writeField params "vocab_only" (->bool v))
       :use-mmap (.writeField params "use_mmap" (->bool v))
       :use-mlock (.writeField params "use_mlock" (->bool v))

       ;; ignore unknown keys
       nil)
     ;; return params
     params)
   (llama_model_default_params)
   m))

;; todo use a soft cache
(defonce ^:private
  token-bufs
  (atom {}))

(defn ^:private get-token-buf [ctx n]
  (get
   (swap! token-bufs
          (fn [m]
            (let [buf (get m ctx)]
              (if (and buf
                       (>= (.size ^Memory buf)
                           (* 4 n)))
                m
                (assoc m ctx (Memory. (* 4 n)))))))
   ctx))


(defn ^:private tokenize* [ctx s add-bos?]
  (let [add-bos (if add-bos?
                  1
                  0)

        s (if add-bos?
            (str " " s)
            s)
        sbytes (.getBytes s "utf-8")
        max-tokens (+ add-bos (alength sbytes))
        token-buf (get-token-buf ctx max-tokens)
        num-tokens (llama_tokenize (:model ctx) sbytes (alength sbytes) token-buf max-tokens add-bos)]
    [num-tokens token-buf]))

(defn get-logits*
  "Returns a copy of the current context's logits as a float array."
  [ctx]
  (let [n-vocab (llama_n_vocab (:model ctx))]
    (-> ^FloatByReference (llama_get_logits ctx)
        .getPointer
        (.getFloatArray 0 n-vocab))))

(defn llama-eval*
  "Adds `s` to the current context and updates the context's logits (see `get-logits`).

  `s`: either be a string or an integer token.
  `n-past`: number of previous tokens to include when updating logits.
  `num-threads`: number of threads to use when updating the logits.
                 If not provided, or `nil`, defaults to `*num-threads*`.
  "
  ([ctx s]
   (llama-eval* ctx s nil))
  ([ctx s n-past]
   (llama-eval* ctx s nil nil))
  ([ctx s n-past num-threads]
   (let [n-past (or n-past (llama_get_kv_cache_token_count ctx))
         [total-tokens ^Memory token-buf]
         (cond
           (string? s)
           (model/tokenize ctx s (zero? n-past))

           (integer? s)
           (let [^Memory buf (get-token-buf ctx 1)]
             [1 (doto buf
                  (.setInt 0 s))]))]
     (assert (< n-past (llama_n_ctx ctx))
             "Context size exceeded")
     (when (and num-threads
                (not= num-threads
                      (:num-threads ctx)))
       (.writeField ^llama_context_params
                    (:context-params ctx)
                    "n_threads"
                    (int num-threads)))

     (let [batch-size (:n-batch ctx)]
       (loop [offset 0
              n-past n-past]
         (let [batch-buf (.share token-buf (* offset 4))
               num-batch-tokens (min batch-size (- total-tokens offset))]
           (llama_eval  ctx batch-buf num-batch-tokens n-past)
           (let [next-offset (+ offset num-batch-tokens)]
             (when (< next-offset total-tokens)
               (recur next-offset
                      (+ n-past num-batch-tokens)))))))

     ctx)))

(def ^:private llama-token-to-piece
  (.getFunction ^com.sun.jna.NativeLibrary libllama
                "llama_token_to_piece"))

(defn decode-token-to-str*
  ([ctx]
   (decode-token-to-str* ctx (Charset/forName "UTF-8")))
  ([ctx opts]
   (let [^Charset charset
         (cond
           (nil? opts) (Charset/forName "UTF-8")
           (map? opts) (or (:charset opts)
                           (Charset/forName "UTF-8"))
           ;; for backwards compatibility
           :else opts)]
     (fn [rf]
       (let [buf-length (int 255)
             buf (Memory. buf-length)
             model (:model ctx)]
         (fn
           ([] (rf))
           ([result] (rf result))
           ([result token]
            (let [nbytes (.invoke llama-token-to-piece
                                  Integer/TYPE
                                  (to-array [model token buf buf-length]))]
              (if (pos? nbytes)
                (rf result (String. (.getByteArray buf 0 nbytes) charset))
                result)))))))))

(def ^:private token-data-size (.size (llama_token_data.)))

(defn ^:private ctx->candidates [ctx candidates-buf*]
  (let [n-vocab (model/n-vocab ctx)
        buf-size (* token-data-size n-vocab)
        candidates-buf @candidates-buf*
        ^Memory
        candidates-buf (if (and candidates-buf
                                (>= (.size ^Memory candidates-buf)
                                    buf-size))
                         candidates-buf
                         (vreset! candidates-buf* (Memory. buf-size)))

        logits (-> ^FloatByReference (llama_get_logits ctx)
                   .getPointer
                   (.getFloatArray 0 n-vocab))]
    (doseq [i (range n-vocab)]
      (let [base-addr (* i token-data-size)
            id i
            logit (aget logits id)
            p 0]
        (.setInt candidates-buf base-addr id)
        (.setFloat candidates-buf (+ base-addr 4) logit)
        (.setFloat candidates-buf (+ base-addr 8) 0)))
    (let [candidates-array-head (doto (Structure/newInstance llama_token_dataByReference
                                                             candidates-buf)
                                  (.read))
          candidates* (doto (llama_token_data_arrayByReference.)
                        (.writeField "data" candidates-array-head)
                        (.writeField "size" (long n-vocab))
                        (.writeField "sorted" (byte 0)))]
      candidates*)))

;; tau default 5.0
;; eta default 0.1
(defn ^:private sample-mirostat-v2* [ctx candidates-buf* mu* tau eta]
  (let [mu (FloatByReference. @mu*)
        candidates (ctx->candidates ctx candidates-buf*)
        next-token (llama_sample_token_mirostat_v2 ctx candidates tau eta mu)]
    (vreset! mu* (.getValue mu))
    next-token))

(defonce ^:private llm-init
  (delay
    (llama_backend_init 0)))

(defn create-context
  "Create and return an opaque llama context."
  ([model-path]
   (create-context model-path nil))
  ([model-path params]
   @llm-init
   (let [
         llama-model-params (map->llama-model-params params)
         model (llama_load_model_from_file model-path llama-model-params)
         _ (when (nil? model)
             (throw (ex-info "Error creating model"
                             {:params params
                              :model-path model-path})))

         ^llama_context_params
         llama-context-params (map->llama-context-params params)
         context (llama_new_context_with_model model llama-context-params)

         ctx-ptr (atom (Pointer/nativeValue context))
         model-ptr (atom (Pointer/nativeValue model))

         model-ref (atom model)
         ;; idempotent cleanup of context
         ;; must not hold references to context!
         delete-context (fn []
                          (let [[old new] (swap-vals! ctx-ptr (constantly nil))]
                            (when old
                              (llama_free (Pointer. old))
                              ;; make sure model doesn't lose
                              ;; all references and get garbage
                              ;; collected until context is freed.
                              (reset! model-ref nil))))
         ;; idempotent cleanup of model
         ;; must not hold references to model!
         delete-model (fn []
                        (let [[old new] (swap-vals! model-ptr (constantly nil))]
                          (when old
                            (llama_free_model (Pointer. old)))))

         n-batch (.readField llama-context-params "n_batch")
         ;; make context autocloseable and implement
         ;; some map lookup interfaces
         context (proxy [Pointer
                         clojure.lang.ILookup
                         java.lang.AutoCloseable
                         com.phronemophobic.llama.impl.model.ILLamaContext]
                     [(Pointer/nativeValue context)]

                   ;; ILLamaContext
                     (token_eos []
                       (llama_token_eos this))
                     (token_bos []
                       (llama_token_bos this))
                     (tokenize [s add-bos?]
                       (tokenize* this s add-bos?))
                     (get_logits []
                       (get-logits* this))
                     (decode_token_to_char
                       ([]
                        (comp cat
                              (decode-token-to-str* this)))
                       ([opts]
                        (comp cat
                              (decode-token-to-str* this opts))))
                     (decode_token_to_str
                       ([]
                        (decode-token-to-str* this))
                       ([opts]
                        (decode-token-to-str* this opts)))
                     (sample_mirostat_v2 [candidates-buf* mu* tau eta]
                       (sample-mirostat-v2* this candidates-buf* mu* tau eta))
                     (set_rng_seed [seed]
                       (llama_set_rng_seed this seed))
                     (n_ctx []
                       (llama_n_ctx this))
                     (n_vocab []
                       (llama_n_vocab (:model this)))
                     (eval
                       ([s]
                        (llama-eval* this s))
                       ([s n-past]
                        (llama-eval* this s n-past))
                       ([s n-past num-threads]
                        (llama-eval* this s n-past num-threads)))

                     (valAt [k]
                       (case k
                         :n-batch n-batch
                         :params params
                         :model @model-ref
                         :context-params llama-context-params
                         :model-params llama-model-params
                         :n-threads (:n-threads params)
                         ;; else
                         nil))
                     (close []
                       (delete-context)
                       (delete-model)))]

     ;; cleanup
     (.register ^Cleaner cleaner context delete-context)
     (.register ^Cleaner cleaner model delete-model)

     context)))

(def llama-model
  (reify
    model/ILLama
    (create-context [_ model-path opts]
      (create-context model-path opts))))
