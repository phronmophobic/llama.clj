(ns com.phronemophobic.llama.raw
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

  ;; private static final int RTLD_LAZY         = 0x00001; /* Lazy function call binding.  */
  ;;   private static final int RTLD_NOW          = 0x00002; /* Immediate function call binding.  */
  ;;   private static final int RTLD_NOLOAD       = 0x00004; /* Do not load the object.  */
;;   private static final int RTLD_DEEPBIND     = 0x00008; /* Use deep binding.  */
 ;; int RTLD_MEMBER = 0x40000; // allows "lib.a(obj.o)" syntax
;; suggestion: https://github.com/java-native-access/jna/issues/724#issuecomment-258937700
;; options.put(Library.OPTION_OPEN_FLAGS,  RTLD_NOW | RTLD_DEEPBIND);
(def ^:private RTLD_LOCAL 0)
(def ^:private RTLD_MEMBER 0x40000)
(def ^:private RTLD_LAZY 0x00001)

(def
  ^java.util.Map
  libllama-options
  {com.sun.jna.Library/OPTION_STRING_ENCODING "UTF8"
   com.sun.jna.Library/OPTION_OPEN_FLAGS (bit-or
                                          RTLD_LOCAL
                                          RTLD_LAZY)})
(def ^:no-doc libllama
  (com.sun.jna.NativeLibrary/getInstance "llama" libllama-options))

(defn ^:private dump-api []
  (let [outf (io/file
              "resources"
              "com"
              "phronemophobic"
              "llama"
              "api.edn")]
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
                      "com/phronemophobic/llama/api.edn"))
                rdr (java.io.PushbackReader. rdr)]
      (edn/read rdr)))

(gen/def-api libllama api)

(let [struct-prefix (gen/ns-struct-prefix *ns*)]
  (defmacro import-structs! []
    `(gen/import-structs! api ~struct-prefix)))

(import-structs!)

(def ^:private llama-token-to-str
  (.getFunction ^com.sun.jna.NativeLibrary libllama
                "llama_token_to_str"))

(defn ^:private preserving-reduced
  [rf]
  #(let [ret (rf %1 %2)]
     (if (reduced? ret)
       (reduced ret)
       ret)))

(defn ^:private decode-token-to-char
  "Returns a transducer that expects a stream of llama tokens
  and outputs a stream of decoded chars.

  The transducer will buffer intermediate results until enough
  bytes to decode a character are available."
  ([ctx]
   (decode-token-to-char ctx nil))
  ([ctx opts]
   (let [^Charset charset
         (cond
           (nil? opts) (Charset/forName "UTF-8")
           (map? opts) (or (:charset opts)
                           (Charset/forName "UTF-8"))
           ;; for backwards compatibility
           :else opts)
         flush? (:flush? opts)]
     (fn [rf]
       (let [decoder (doto (.newDecoder charset)
                       (.onMalformedInput CodingErrorAction/REPLACE)
                       (.onUnmappableCharacter CodingErrorAction/REPLACE))

             input-buffer (ByteBuffer/allocate 256)
             output-buffer (CharBuffer/allocate 256)

             rrf (preserving-reduced rf)]
         (fn
           ([] (rf))
           ([result]
            (if flush?
              (do
                (.flip input-buffer)
                (let [result
                      (let [ ;; Invoke the decode method one final time, passing true for the endOfInput argument; and then
                            decoder-result1 (.decode decoder input-buffer output-buffer true)
                            ;; Invoke the flush method so that the decoder can flush any internal state to the output buffer.
                            decoder-result2 (.flush decoder output-buffer)]
                        (if (and (.isUnderflow decoder-result1)
                                 (.isUnderflow decoder-result2))
                          (do
                            (.flip output-buffer)
                            (let [result (reduce rrf result output-buffer)]
                              (.clear output-buffer)
                              result))
                          ;; else
                          (throw (Exception. "Unexpected decoder state."))))]
                  (rf result)))
              ;; else no flush
              (rf result)))
           ([result token]
            (let [^Pointer p (.invoke
                              ^com.sun.jna.Function llama-token-to-str
                              Pointer (to-array [ctx (int token)]))
                  ;; p points to a c string
                  ;; find length by counting until null token is found
                  len (loop [i 0]
                        (if (zero? (.getByte p i))
                          i
                          (recur (inc i))))]
              (.put input-buffer (.getByteBuffer p 0 len))
              (.flip input-buffer)

              ;; Invoke the decode method zero or more times, as long as additional input may be available, passing false for the endOfInput argument and filling the input buffer and flushing the output buffer between invocations;
              (let [decoder-result (.decode decoder input-buffer output-buffer false)]
                (cond
                  (.isUnderflow decoder-result)
                  (do
                    (.compact input-buffer)
                    (.flip output-buffer)
                    (let [result (reduce rrf result output-buffer)]
                      (.clear output-buffer)
                      result))

                  (.isOverflow decoder-result)
                  (throw (ex-info "Decoder buffer too small" {}))

                  (.isError decoder-result)
                  (throw (ex-info "Decoder Error" {:decoder decoder}))

                  :else
                  (throw (Exception. "Unexpected decoder state."))))))))))))



(defn ^:private char->str
  "Transducer that expects a stream of chars. If a surrogate pair is detected,
  wait until the full pair is available before emitting."
  []
  (fn [rf]
    (let [v (volatile! nil)]
      (fn
        ([] (rf))
        ([result]
         (let [result (if-let [c @v]
                        (unreduced (rf result c))
                        result)]
           (rf result)))
        ([result c]
         (if-let [c1 @v]
           (do
             (vreset! v nil)
             (rf result (str c1 c)))
           (if (Character/isHighSurrogate c)
             (do
               (vreset! v c)
               result)
             (rf result (str c)))))))))

(defn ^:private decode-token
  "Returns a transducer that expects a stream of llama tokens
  and outputs a stream of strings.

  The transducer will buffer intermediate results until enough
  bytes to decode a character are available. Also combines
  surrogate pairs of characters."
  ([ctx]
   (decode-token ctx (Charset/forName "UTF-8")))
  ([ctx ^Charset charset]
   (comp
    (decode-token-to-char ctx charset)
    (char->str))))


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

(defn ^:private tokenize* [ctx ^String s add-bos?]
  (let [add-bos (if add-bos?
                  1
                  0)

        s (if add-bos?
            (str " " s)
            s)
        max-tokens (+ add-bos (alength (.getBytes s "utf-8")))
        token-buf (get-token-buf ctx max-tokens)
        num-tokens (llama_tokenize ctx s token-buf max-tokens add-bos)]
    [num-tokens token-buf]))

(def ^:dynamic
  *num-threads*
  "Number of threads used when generating tokens."
  (.. Runtime getRuntime availableProcessors))

(defn ^:private llama-eval*
  "Adds `s` to the current context and updates the context's logits (see `get-logits`).

  `s`: either be a string or an integer token.
  `n-past`: number of previous tokens to include when updating logits.
  `num-threads`: number of threads to use when updating the logits.
                 If not provided, or `nil`, defaults to `*num-threads*`.
  "
  ([ctx s]
   (llama-eval* ctx s nil *num-threads*))
  ([ctx s n-past]
   (llama-eval* ctx s n-past *num-threads*))
  ([ctx s n-past num-threads]
   (let [num-threads (or num-threads *num-threads*)
         n-past (or n-past
                    (llama_get_kv_cache_token_count ctx))
         [total-tokens ^Memory token-buf]
         (cond
           (string? s)
           (tokenize* ctx s (zero? n-past))

           (integer? s)
           (let [^Memory buf (get-token-buf ctx 1)]
             [1 (doto buf
                  (.setInt 0 s))]))]
     (assert (< n-past (llama_n_ctx ctx))
             "Context size exceeded")

     (let [batch-size (:n-batch ctx)]
       (loop [offset 0
              n-past n-past]
         (let [batch-buf (.share token-buf (* offset 4))
               num-batch-tokens (min batch-size (- total-tokens offset))]
           (llama_eval ctx batch-buf num-batch-tokens n-past num-threads)
           (let [next-offset (+ offset num-batch-tokens)]
             (when (< next-offset total-tokens)
               (recur next-offset
                      (+ n-past num-batch-tokens)))))))

     ctx)))

(def ^:private token-data-size (.size (llama_token_data.)))

(defn ^:private ctx->candidates [ctx candidates-buf*]
  (let [n-vocab (llama_n_vocab ctx)
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

(defn ^:private get-embedding*
  [ctx]
  (let [^com.sun.jna.ptr.FloatByReference
        fbr (llama_get_embeddings ctx)
        p (.getPointer fbr)
        arr (float-array
             (llama_n_embd ctx))]
    (.read p 0 arr 0 (alength arr))
    arr))

(defn ^:private get-logits*
  "Returns a copy of the current context's logits as a float array."
  [ctx]
  (let [n-vocab (llama_n_vocab ctx)]
    (-> ^FloatByReference (llama_get_logits ctx)
        .getPointer
        (.getFloatArray 0 n-vocab))))



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

(defn ^:private map->llama-params [m]
  (reduce-kv
   (fn [^llama_context_params
        params k v]
     (case k
       :seed (.writeField params "seed" (int v))
       :n-ctx (.writeField params "n_ctx" (int v))
       :n-batch (.writeField params "n_batch" (int v))
       :n-gpu-layers (.writeField params "n_gpu_layers" (int v))
       :main-gpu (.writeField params "main_gpu" (int v))
       :tensor-split (.writeField params "tensor_split" (->float-array-by-reference  v))
       :rope-freq-base (.writeField params "rope_freq_base" (float v))
       :rope-freq-scale (.writeField params "rope_freq_scale" (float v))
       ;; :progress-callback (.writeField params "progress_callback" v)
       ;; :progress-callback-user-data (.writeField params "progress_callback_user_data" v)
       :low-vram (.writeField params "low_vram" (->bool v))
       :mul_mat_q (.writeField params "mul_mat_q" (->bool v))
       :f16-kv (.writeField params "f16_kv" (->bool v))
       :logits-all (.writeField params "logits_all" (->bool v))
       :vocab-only (.writeField params "vocab_only" (->bool v))
       :use-mmap (.writeField params "use_mmap" (->bool v))
       :use-mlock (.writeField params "use_mlock" (->bool v))
       :embedding (.writeField params "embedding" (->bool v))
       :gqa (.writeField params "n_gqa" (int v))
       :rms-norm-eps (.writeField params "rms_norm_eps" (float v))
       ;; else ignore
       params)
     ;; return params
     params)
   (llama_context_default_params)
   m))

(defonce ^:private llm-init
  (delay
    (llama_backend_init 0)))

(defn ^:private create-context
  "Create and return an opaque llama context.

  `model-path` should be an absolute or relative path to a F16, Q4_0, Q4_1, Q5_0, Q5_1, or Q8_0 ggml model.

  An optional map of parameters may be passed for parameterizing the model. The following keys map to their corresponding llama.cpp equivalents:
  - `:seed`: RNG seed, -1 for random
  - `:n-ctx`: text context
  - `:n-batch`: prompt processing batch size
  - `:n-gpu-layers`: number of layers to store in VRAM
  - `:main-gpu`: the GPU that is used for scratch and small tensors
  - `:tensor-split`: how to split layers across multiple GPUs
  - `:rope-freq-base`: RoPE base frequency
  - `:rope-freq-scale`: RoPE frequency scaling factor
  - `:low-vram`: if true, reduce VRAM usage at the cost of performance
  - `:mul_mat_q`: if true, use experimental mul_mat_q kernels
  - `:f16-kv`: use fp16 for KV cache
  - `:logits-all`: the llama_eval() call computes all logits, not just the last one
  - `:vocab-only`: only load the vocabulary, no weights
  - `:use-mmap`: use mmap if possible
  - `:use-mlock`: force system to keep model in RAM
  - `:embedding`: embedding mode only
  - `:gqa`: grouped-query attention factor (TEMP!!! use 8 for LLaMAv2 70B)
  - `:rms-norm-eps`: rms norm eps (TEMP!!! use 1e-5 for LLaMAv2)

  Resources can be freed by calling .close on the returned context.
  Using a closed context is undefined and will probably crash the JVM.

  Contexts are not thread-safe. Using the same context on multiple threads
  is undefined and will probably crash the JVM.
  "
  ([model-path]
   (create-context model-path nil))
  ([model-path
    {:keys [seed
            n-ctx
            n-batch
            n-gpu-layers
            main-gpu
            tensor-split
            rope-freq-base
            rope-freq-scale
            low-vram
            f16-kv
            logits-all
            vocab-only
            use-mmap
            use-mlock
            embedding
            gqa
            rms-norm-eps]
     :as params}]
   @llm-init
   (let [
         ;; llama-model-params (map->llama-model-params params)
         ;; model (llama_load_model_from_file model-path llama-model-params)
         ;; _ (when (nil? model)
         ;;     (throw (ex-info "Error creating model"
         ;;                     {:params params
         ;;                      :model-path model-path})))

         ;; ^llama_context_params
         ;; llama-context-params (map->llama-context-params params)
         ;; context (llama_new_context_with_model model llama-context-params)

         ^llama_context_params
         llama-params (map->llama-params params)
         model (llama_load_model_from_file model-path llama-params)
         _ (when (nil? model)
             (throw (ex-info "Error creating model"
                             {:params params
                              :model-path model-path})))
         context (llama_new_context_with_model model llama-params)         

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

         n-batch (.readField llama-params "n_batch")

         eos (llama_token_eos)
         ;; make context autocloseable and implement
         ;; some map lookup interfaces
         context (proxy [Pointer
                         clojure.lang.ILookup
                         java.lang.AutoCloseable
                         com.phronemophobic.llama.impl.model.ILLamaContext]
                     [(Pointer/nativeValue context)]

                   ;; ILLamaContext
                     (token_eos []
                       eos)
                     (token_bos []
                       (llama_token_bos))
                     (token_is_eog [token]
                       (= eos token))
                     (tokenize [s add-bos?]
                       (tokenize* this s add-bos?))
                     (get_embedding []
                       (get-embedding* this))
                     (get_logits []
                       (get-logits* this))
                     (decode_token_to_char
                       ([]
                        (decode-token-to-char this))
                       ([opts]
                        (decode-token-to-char this opts)))
                     (decode_token_to_str
                       ([]
                        (decode-token this))
                       ([opts]
                        (decode-token this opts)))
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
                         :model-format :ggml
                         :impl ::impl
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

