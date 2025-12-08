(ns com.phronemophobic.llama.raw-gguf-b7325
  (:require [com.phronemophobic.clong.gen.jna :as gen]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [com.phronemophobic.llama.raw-gguf-lib :as libllama]
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
           com.sun.jna.Structure
           com.sun.jna.Platform)
  (:gen-class))

(def cleaner (delay (Cleaner/create)))

(defn ^:private random-int []
  (.nextInt (java.util.Random.)))

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

(def default-arguments
  [ "-resource-dir"
 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/15.0.0"
 "-isysroot"
 "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
 "-I/usr/local/include"
 "-internal-isystem"
 "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/local/include"
 "-internal-isystem"
 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/15.0.0/include"
 "-internal-externc-isystem"
 "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include"
 "-internal-externc-isystem"
 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include"
])

(defn ^:private dump-api []
  (let [outf (io/file
              "resources"
              "com"
              "phronemophobic"
              "llama"
              "api-gguf-b7325.edn")]
    (.mkdirs (.getParentFile outf))
    (with-open [w (io/writer outf)]
      (write-edn w
                 ((requiring-resolve 'com.phronemophobic.clong.clang/easy-api)
                  ;; "/Users/adrian/workspace/llama.cpp/llama.h"
                  "/Users/adrian/workspace/cljonda/build/llama.cpp/include/llama.h"
                  (into default-arguments
                        ["-I/Users/adrian/workspace/cljonda/build/llama.cpp/ggml/include"]))))))

(def raw-api
  #_((requiring-resolve 'com.phronemophobic.clong.clang/easy-api) "/Users/adrian/workspace/llama.cpp/llama.h")
  (with-open [rdr (io/reader
                     (io/resource
                      "com/phronemophobic/llama/api-gguf-b7325.edn"))
                rdr (java.io.PushbackReader. rdr)]
      (edn/read rdr)))

;; there's a struct and a function named ggml_backend_graph_copy
;; which causes problems.
(defn remove-broken [api]
  (specter/setval
   [:functions
    specter/ALL
    #(= :ggml_backend_graph_copy
        (:id %))]
   specter/NONE
   api))

(defn adjust-batch-struct
  "The llama_batch struct uses an int pointer for the token field.
  That gets translated to IntByReference in JNA which is annoying to work with."
  [api]
  (specter/setval
   [:structs
    specter/ALL
    #(= :clong/llama_batch (:id %))
    :fields
    specter/ALL
    #(#{"token" "pos"} (:name %))
    :datatype]
   :coffi.mem/pointer
   api))

(defn adjust-chat-message-struct
  "The llama_batch struct uses an char pointer for the token field.
  That gets translated to ByteByReference in JNA which is annoying to work with."
  [api]
  (specter/setval
   [:structs
    specter/ALL
    #(= :clong/llama_chat_message (:id %))
    :fields
    specter/ALL
    #(#{"role" "content"} (:name %))
    :datatype]
   String
   api))

(def api (-> raw-api
             (remove-broken)
             (adjust-batch-struct)
             (adjust-chat-message-struct)))

(gen/def-api-lazy libllama/libllama api)

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
   (fn [^Structure
        params k v]
     (case k
       :n-ctx (.writeField params "n_ctx" (int v))
       :n-batch (.writeField params "n_batch" (int v))
       :n-ubatch (.writeField params "n_ubatch" (int v))
       :n-threads (.writeField params "n_threads" (int v))
       :n-threads-batch (.writeField params "n_threads_batch" (int v))

       :rope-freq-base (.writeField params "rope_freq_base" (float v))
       :rope-freq-scale (.writeField params "rope_freq_scale" (float v))
       :yarn-ext-factor (.writeField params "yarn_ext_factor" (float v))
       :yarn-attn-factor (.writeField params "yarn_attn_factor" (float v))
       :yarn-beta-fast (.writeField params "yarn_beta_fast" (float v))
       :yarn-beta-slow (.writeField params "yarn_beta_slow" (float v))
       :yarn-orig-ctx (.writeField params "yarn_orig_ctx" (float v))
       :defrag-thold (.writeField params "defrag_thold" (float v))

       :logits-all (.writeField params "logits_all" (->bool v))
       ;; for backwards compatibility.
       ;; the embedding param was renamed to embeddings
       :embedding (.writeField params "embeddings" (->bool v))
       :embeddings (.writeField params "embeddings" (->bool v))

       :offload-kqv (.writeField params "offload_kqv" (->bool v))
       :flast-attn (.writeField params "flast_attn" (->bool v))
       :no-perf (.writeField params "no_perf" (->bool v))

       ;; ignore unknown keys
       nil)
     ;; return params
     params)
   (llama_context_default_params)
   m))

(defn ^:private map->llama-model-params [m]
  (reduce-kv
   (fn [^Structure
        params k v]
     (case k
       :n-gpu-layers (.writeField params "n_gpu_layers" (int v))
       :main-gpu (.writeField params "main_gpu" (int v))
       :tensor-split (.writeField params "tensor_split" (->float-array-by-reference  v))

       :vocab-only (.writeField params "vocab_only" (->bool v))
       :use-mmap (.writeField params "use_mmap" (->bool v))
       :use-mlock (.writeField params "use_mlock" (->bool v))
       :check-tensors (.writeField params "check_tensors" (->bool v))

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


(defn ^:private tokenize* [ctx ^String s add-bos?]
  (let [add-bos (if add-bos?
                  1
                  0)
        sbytes (.getBytes s "utf-8")
        max-tokens (+ add-bos 1 (alength sbytes))
        token-buf (get-token-buf ctx max-tokens)
        num-tokens (llama_tokenize (:vocab ctx) sbytes (alength sbytes) token-buf max-tokens add-bos 1)]
    [num-tokens token-buf]))

(defn ^:private get-embedding*
  ([ctx]
   (let [^com.sun.jna.ptr.FloatByReference
         fbr
         (or
          ;; only support seq id 0 right now
          (llama_get_embeddings_seq ctx 0)
          (llama_get_embeddings_ith ctx
                                    -1))
         p (.getPointer fbr)
         arr (float-array
              (llama_model_n_embd (:model ctx)))]
     (.read p 0 arr 0 (alength arr))
     arr)))

(defn ^:private get-logits*
  "Returns a copy of the current context's logits as a float array."
  [ctx]
  (let [n-vocab (llama_vocab_n_tokens (:vocab ctx))]
    (-> ^FloatByReference (llama_get_logits_ith ctx -1)
        .getPointer
        (.getFloatArray 0 n-vocab))))

(defn ^:private llama-eval*
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
   (let [
         ;; need to keep track of n-past ourselves now.
         _ (when n-past
             (llama_memory_seq_rm (llama_get_memory ctx) 0 n-past -1))
         n-past (or n-past @(:n-past ctx))

         [total-tokens ^Memory token-buf]
         (cond
           (string? s)
           (model/tokenize ctx s (zero? n-past))

           (integer? s)
           (let [^Memory buf (get-token-buf ctx 1)]
             [1 (doto buf
                  (.setInt 0 s))]))]
     (when (not (< n-past (llama_n_ctx ctx)))
       (throw (ex-info "Context size exceeded."
                       {})))
     (when (and num-threads
                (not= num-threads
                      (:num-threads ctx)))
       (.writeField ^Structure
                    (:context-params ctx)
                    "n_threads"
                    (int num-threads)))

     (let [batch-size (:n-batch ctx)
           pos-buf (Memory. (* 4 batch-size))
           batch (doto (llama_batch.)
                   (.writeField "embd" nil)
                   (.writeField "n_seq_id" nil)
                   (.writeField "seq_id" nil)
                   (.writeField "logits" nil)
                   (.writeField "pos" pos-buf))]
       (loop [offset 0
              n-past n-past]
         (let [batch-buf (.share token-buf (* offset 4))
               ^long
               num-batch-tokens (min batch-size (- total-tokens offset))]

           (.writeField batch "n_tokens" (int (min batch-size (- total-tokens offset))))
           (.writeField batch "token" batch-buf)
           (.write pos-buf 0 (int-array (range n-past (+ n-past num-batch-tokens))) 0 num-batch-tokens)

           (llama_decode ctx batch)

           (let [next-offset (+ offset num-batch-tokens)]
             (when (< next-offset total-tokens)
               (recur next-offset
                      (+ n-past num-batch-tokens))))))
       (reset! (:n-past ctx) (+ n-past total-tokens)))
     ctx)))

(defn ^:private decode-token-to-buf
  ([ctx]
   (fn [rf]
     (let [buf-length (int 255)
           buf (Memory. buf-length)
           vocab (:vocab ctx)
           skip-whitespace* (volatile! true)]
       (fn
         ([] (rf))
         ([result] (rf result))
         ([result token]
          (let [nbytes (llama_token_to_piece vocab
                                             token
                                             buf
                                             buf-length
                                             0
                                             1
                                             )]
            (if (pos? nbytes)
              (let [bb
                    (if @skip-whitespace*
                      (do
                        (vreset! skip-whitespace* false)
                        (if (= 32 (.getByte buf 0))
                          (let [len (dec nbytes)]
                            (when (pos? len)
                              (.getByteBuffer buf 1 (dec nbytes))))
                          (.getByteBuffer buf 0 nbytes)))
                      (.getByteBuffer buf 0 nbytes))]
                (when bb
                  (rf result bb)))
              result))))))))

(comment
  (def buf (Memory. 255))
  (def model (:model com.phronemophobic.llama/ctx))
  (def toks [5255])
  (def dog 3914)
  (def happy [28705 30464])
  (def nbytes (.invoke llama-token-to-piece
                       Integer/TYPE
                       (to-array [model 28705 buf 255])))

  (def nbytes (.invoke llama-token-to-piece
                       Integer/TYPE
                       (to-array [model 30464 buf 255])))
  (into [](.getByteArray buf 0 nbytes))
  (String. (.getByteArray buf 0 nbytes))
  ,)

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
     (comp
      (decode-token-to-buf ctx)
      (fn [rf]
        (let [model (:model ctx)

              decoder (doto (.newDecoder charset)
                        (.onMalformedInput CodingErrorAction/REPLACE)
                        (.onUnmappableCharacter CodingErrorAction/REPLACE))

              input-buffer (ByteBuffer/allocate 256)
              output-buffer (CharBuffer/allocate 256)

              buf-length (int 255)
              buf (Memory. buf-length)
              skip-whitespace* (volatile! true)

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
            ([result ^java.nio.ByteBuffer bb]
             (.put input-buffer bb)
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

(defn ^:private decode-token-to-str*
  "Returns a transducer that expects a stream of llama tokens
  and outputs a stream of strings.

  The transducer will buffer intermediate results until enough
  bytes to decode a character are available. Also combines
  surrogate pairs of characters."
  ([ctx]
   (decode-token-to-str* ctx (Charset/forName "UTF-8")))
  ([ctx ^Charset charset]
   (comp
    (decode-token-to-char ctx charset)
    (char->str))))


(defn ^:private chat-template [ctx ^Memory buf]
  (when-not ctx
    (throw (IllegalArgumentException. "No chat template found for context.")))
  (llama_model_chat_template (:model ctx) nil))

(def ^:private message-size (delay
                              (.size (llama_chat_message.))))
(defn chat-apply-template* [ctx-or-template messages opts]
  (let [append-start-assistant-message? (get opts :append-start-assistant-message? true)
        content-length (transduce
                        (comp (map :content)
                              (map count))
                        +
                        messages)
        buf (Memory. (max 2048
                          content-length))
        template (if (string? ctx-or-template)
                   ctx-or-template
                   (chat-template ctx-or-template buf))]
    (if (not template)
      (throw (IllegalArgumentException. "No chat template found for context."))

      ;; else
      (let [messages* (Memory. (* @message-size
                                  (count messages)))]
        (doseq [[i msg] (map-indexed vector messages)]
          (let [msg* (Structure/newInstance llama_chat_messageByReference (.share messages* (* i @message-size)))]
            (.writeField msg* "role" (:role msg))
            (.writeField msg* "content" (:content msg))))
        (loop [buf buf]
          (let [ret (llama_chat_apply_template template
                                               messages*
                                               (count messages)
                                               (if append-start-assistant-message?
                                                 1
                                                 0)
                                               buf
                                               (.size buf))]
            (cond
              (neg? ret) (throw (IllegalArgumentException. "No chat template found for context."))
              (<= ret (.size buf)) (.getString buf 0 "utf-8")
              :else (recur (Memory. (inc ret))))))))))

(defn metadata* [ctx]
  (let [model (:model ctx)
        n (llama_model_meta_count model)]
    ;; keys and values seem to be much shorter than 256
    ;; can revisit if things change.
    (loop [buf (Memory. 256)
           i 0
           meta {}]
      (if (>= i n)
        meta
        (let [_written (llama_model_meta_key_by_index model i buf (.size buf))
              k (.getString buf 0 "utf-8")
              _written (llama_model_meta_val_str_by_index model i buf (.size buf))
              v (.getString buf 0 "utf-8")]
          (recur buf
                 (inc i)
                 (assoc meta k v)))))))

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

(defn ^:private init-mirostat-v2-sampler*
  ([ctx tau eta]
   (let [sampler* (::sampler ctx)
         sampler-seed* (::sampler-seed ctx)

         samplef
         (fn [_logits]
           (let [
                 next-seed @sampler-seed*
                 previous-sampler @sampler*

                 ;; create a new sampler if the seed has changed
                 ;; or we don't have an existing sampler
                 ;; set a new seed to create a new sampler
                 sampler (if (or (nil? previous-sampler)
                                 next-seed)
                           (let [sparams (llama_sampler_chain_default_params)
                                 ^llama_sampler
                                 sampler (llama_sampler_chain_init sparams)
                                 ptr (Pointer/nativeValue (.getPointer  sampler))
                                 _ (.register ^Cleaner @cleaner
                                              sampler
                                              (fn []
                                                (llama_sampler_free (Pointer. ptr))))

                                 seed (or next-seed (random-int))
                                 _ (llama_sampler_chain_add sampler (llama_sampler_init_mirostat_v2 seed tau eta))]
                             ;; ignore race conditions
                             ;; generating from multiple threads doesn't work anyway
                             (reset! sampler-seed* nil)
                             (reset! sampler* sampler)

                             sampler)
                           previous-sampler)
                 token (llama_sampler_sample sampler ctx -1)]
             (llama_sampler_accept sampler token)
             token))]
     samplef)))

(defonce ^:private llm-init
  (delay
    (llama_backend_init)))

(defn ^:private create-context
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
         vocab (llama_model_get_vocab model)

         ^llama_context_params
         llama-context-params (map->llama-context-params params)
         context (llama_init_from_model model llama-context-params)

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

         sampler (atom nil)
         sampler-seed (atom (let [params-seed (:seed params)]
                              (cond
                                (= -1 params-seed) (random-int)
                                (nil? params-seed) (random-int)
                                :else (int params-seed))))

         n-batch (.readField llama-context-params "n_batch")
         n-past (atom 0)
         ;; make context autocloseable and implement
         ;; some map lookup interfaces
         context (proxy [Pointer
                         clojure.lang.ILookup
                         java.lang.AutoCloseable
                         com.phronemophobic.llama.impl.model.ILLamaContext]
                     [(Pointer/nativeValue context)]

                   ;; ILLamaContext
                     (token_eos []
                       (llama_token_eos (:vocab this)))
                     (token_bos []
                       (llama_token_bos (:vocab this)))
                     (token_is_eog [token]
                       (not (zero? (llama_token_is_eog (:vocab this) token))))
                     (tokenize [s add-bos?]
                       (tokenize* this s add-bos?))
                     (get_embedding []
                       (get-embedding* this))
                     (get_logits []
                       (get-logits* this))
                     (decode_token_to_char
                       ([]
                        (comp (decode-token-to-str* this)
                              cat))
                       ([opts]
                        (comp (decode-token-to-str* this opts)
                              cat)))
                     (decode_token_to_str
                       ([]
                        (decode-token-to-char this))
                       ([opts]
                        (decode-token-to-char this opts)))
                     (metadata []
                       (metadata* this))
                     (model_description []
                       (let [buf (Memory. 512)]
                         (llama_model_desc (:model this) buf (.size buf))
                         (.getString buf 0 "utf-8")))
                     (model_size []
                       (llama_model_size (:model this)))
                     (model_n_params []
                       (llama_model_n_params (:model this)))
                     (init_mirostat_v2_sampler [tau eta]
                       (init-mirostat-v2-sampler* this tau eta))
                     ;; Deprecated
                     #_(sample_mirostat_v2 [candidates-buf* mu* tau eta])
                     (set_rng_seed [seed]
                       (reset! sampler-seed seed))
                     (n_ctx []
                       (llama_n_ctx this))
                     (n_vocab []
                       (llama_vocab_n_tokens (:vocab this)))
                     (n_embd []
                       (llama_model_n_embd (:model this)))
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
                         :model-format :gguf
                         :impl ::impl
                         :n-past n-past
                         :vocab vocab
                         ;; used by mirostat sampler
                         ;; may be used by more generic sampling
                         ;; in the future
                         ::sampler-seed sampler-seed
                         ::sampler sampler

                         ;; else
                         nil))
                     (close []
                       (delete-context)
                       (delete-model)))]

     ;; cleanup
     (.register ^Cleaner @cleaner context delete-context)
     (.register ^Cleaner @cleaner model delete-model)

     context)))

(def llama-model
  (reify
    model/ILLama
    (create-context [_ model-path opts]
      (create-context model-path opts))
    (set_log_callback [_ cb]
      (llama_log_set cb nil))
    (chat_apply_template [this template messages opts]
      (chat-apply-template* template messages opts))))


