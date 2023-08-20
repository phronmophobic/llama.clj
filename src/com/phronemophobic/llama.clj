(ns com.phronemophobic.llama
  (:require [com.phronemophobic.llama.raw :as raw]
            [clojure.string :as str])
  (:import java.lang.ref.Cleaner
           java.nio.charset.CodingErrorAction
           java.nio.charset.CharsetDecoder
           java.nio.charset.Charset
           java.nio.ByteBuffer
           java.nio.CharBuffer
           com.sun.jna.Memory
           com.sun.jna.Pointer
           com.sun.jna.ptr.IntByReference
           com.sun.jna.ptr.FloatByReference
           com.sun.jna.Structure))

(raw/import-structs!)
(defonce cleaner (delay (Cleaner/create)))

(def ^:dynamic
  *num-threads*
  "Number of threads used when generating tokens."
  (.. Runtime getRuntime availableProcessors))

(def ^:private token-data-size (.size (llama_token_data.)))

(defn eos
  "Returns the llama end of sentence token."
  []
  (raw/llama_token_eos))
(defn bos
  "Returns the llama beginning of sentence token."
  []
  (raw/llama_token_bos))

(defonce ^:private llm-init
  (delay
    (raw/llama_backend_init 0)))

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
       :rms-norm-eps (.writeField params "rms_norm_eps" (float v)))
     ;; return params
     params)
   (raw/llama_context_default_params)
   m))

(defn create-context
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
   (let [^llama_context_params
         llama-params (map->llama-params params)
         model (raw/llama_load_model_from_file model-path llama-params)
         _ (when (nil? model)
             (throw (ex-info "Error creating model"
                             {:params params
                              :model-path model-path})))
         context (raw/llama_new_context_with_model model llama-params)

         ctx-ptr (atom (Pointer/nativeValue context))
         model-ptr (atom (Pointer/nativeValue model))

         model-ref (atom model)
         ;; idempotent cleanup of context
         ;; must not hold references to context!
         delete-context (fn []
                          (let [[old new] (swap-vals! ctx-ptr (constantly nil))]
                            (when old
                              (raw/llama_free (Pointer. old))
                              ;; make sure model doesn't lose
                              ;; all references and get garbage
                              ;; collected until context is freed.
                              (reset! model-ref nil))))
         ;; idempotent cleanup of model
         ;; must not hold references to model!
         delete-model (fn []
                        (let [[old new] (swap-vals! model-ptr (constantly nil))]
                          (when old
                            (raw/llama_free_model (Pointer. old)))))

         n-batch (.readField llama-params "n_batch")
         ;; make context autocloseable and implement
         ;; some map lookup interfaces
         context (proxy [Pointer
                         clojure.lang.ILookup
                         java.lang.AutoCloseable]
                     [(Pointer/nativeValue context)]
                   (valAt [k]
                     (case k
                       :n-batch n-batch
                       :params params
                       :model @model-ref
                       ;; else
                       nil))
                   (close []
                     (delete-context)
                     (delete-model)))]

     ;; cleanup
     (.register ^Cleaner @cleaner context delete-context)
     (.register ^Cleaner @cleaner model delete-model)

     context)))

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

(defn ^:private tokenize [ctx s add-bos?]
  (let [add-bos (if add-bos?
                  1
                  0)

        s (if add-bos?
            (str " " s)
            s)
        max-tokens (+ add-bos (count s))
        token-buf (get-token-buf ctx max-tokens)
        num-tokens (raw/llama_tokenize ctx s token-buf max-tokens add-bos)]
    [num-tokens token-buf]))

(defn llama-update
  "Adds `s` to the current context and updates the context's logits (see `get-logits`).

  `s`: either be a string or an integer token.
  `n-past`: number of previous tokens to include when updating logits.
  `num-threads`: number of threads to use when updating the logits.
                 If not provided, or `nil`, defaults to `*num-threads*`.
  "
  ([ctx s]
   (llama-update ctx s (raw/llama_get_kv_cache_token_count ctx) *num-threads*))
  ([ctx s n-past]
   (llama-update ctx s n-past *num-threads*))
  ([ctx s n-past num-threads]
   (let [num-threads (or num-threads *num-threads*)
         [total-tokens ^Memory token-buf]
         (cond
           (string? s)
           (tokenize ctx s (zero? n-past))

           (integer? s)
           (let [^Memory buf (get-token-buf ctx 1)]
             [1 (doto buf
                  (.setInt 0 s))]))]
     (assert (< n-past (raw/llama_n_ctx ctx))
             "Context size exceeded")

     (let [batch-size (:n-batch ctx)]
       (loop [offset 0
              n-past n-past]
         (let [batch-buf (.share token-buf (* offset 4))
               num-batch-tokens (min batch-size (- total-tokens offset))]
           (raw/llama_eval ctx batch-buf num-batch-tokens n-past num-threads)
           (let [next-offset (+ offset num-batch-tokens)]
             (when (< next-offset total-tokens)
               (recur next-offset
                      (+ n-past num-batch-tokens)))))))

     ctx)))

(defn sample-logits-greedy
  "Returns the token with the highest value.

  `logits`: a collection of floats representing the logits (see `get-logits`)."
  [logits]
  (transduce (map-indexed vector)
             (completing
              (fn [[idx1 f1 :as r1] [idx2 f2 :as r2]]
                (if (> f1 f2)
                  r1
                  r2))
              first)
             [nil Float/MIN_VALUE]
             logits))

(defn ^:private ctx->candidates [ctx candidates-buf*]
  (let [n-vocab (raw/llama_n_vocab ctx)
        buf-size (* token-data-size n-vocab)
        candidates-buf @candidates-buf*
        ^Memory
        candidates-buf (if (and candidates-buf
                                (>= (.size ^Memory candidates-buf)
                                    buf-size))
                         candidates-buf
                         (vreset! candidates-buf* (Memory. buf-size)))

        logits (-> ^FloatByReference (raw/llama_get_logits ctx)
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
(defn ^:private sample-mirostat-v2 [ctx candidates-buf* mu* tau eta]
  (let [mu (FloatByReference. @mu*)
        candidates (ctx->candidates ctx candidates-buf*)
        next-token (raw/llama_sample_token_mirostat_v2 ctx candidates tau eta mu)]
    (vreset! mu* (.getValue mu))
    next-token))

(defn init-mirostat-v2-sampler
  "Given a context, returns a sampling function that uses the llama.cpp mirostat_v2 implementation."
  ([ctx]
   (let [tau (float 5.0)
         eta (float 0.1)]
     (init-mirostat-v2-sampler ctx tau eta)))
  ([ctx tau eta]
   (fn [logits]
     (sample-mirostat-v2 ctx
                         (volatile! nil)
                         (volatile! (* 2 tau))
                         tau
                         eta))))

(defn get-logits
  "Returns a copy of the current context's logits as a float array."
  [ctx]
  (let [n-vocab (raw/llama_n_vocab ctx)]
    (-> ^FloatByReference (raw/llama_get_logits ctx)
        .getPointer
        (.getFloatArray 0 n-vocab))))


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

(defn ^:private preserving-reduced
  [rf]
  #(let [ret (rf %1 %2)]
     (if (reduced? ret)
       (reduced ret)
       ret)))

(def ^:private llama-token-to-str
  (.getFunction ^com.sun.jna.NativeLibrary raw/libllama
                "llama_token_to_str"))

(defn decode-token-to-char
  "Returns a transducer that expects a stream of llama tokens
  and outputs a stream of decoded chars.

  The transducer will buffer intermediate results until enough
  bytes to decode a character are available."
  ([ctx]
   (decode-token-to-char ctx (Charset/forName "UTF-8")))
  ([ctx ^Charset charset]
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
                (throw (Exception. "Unexpected decoder state.")))))))))))


(defn decode-token
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

(defn generate-tokens
  "Returns a seqable/reducible sequence of tokens from ctx with prompt."
  ([ctx prompt]
   (generate-tokens ctx prompt nil))
  ([ctx prompt {:keys [samplef
                       num-threads
                       seed
                       ;; resize-context
                       ]
                :as opts}]
   (let [samplef (or samplef
                     (init-mirostat-v2-sampler ctx))
         eos (raw/llama_token_eos)]
     (reify
       clojure.lang.Seqable
       (seq [_]
         (when seed
           (raw/llama_set_rng_seed ctx seed))
         ((fn next [ctx]
            (let [next-token (samplef (get-logits ctx))]
              (when (not= eos next-token)
                (cons next-token
                      (lazy-seq (next (llama-update ctx next-token (raw/llama_get_kv_cache_token_count ctx) num-threads)))))))
          (llama-update ctx prompt 0 num-threads)))
       clojure.lang.IReduceInit
       (reduce [_ rf init]
         (when seed
           (raw/llama_set_rng_seed ctx seed))
         (loop [acc init
                ret (llama-update ctx prompt 0 num-threads)]
           (let [next-token (samplef (get-logits ctx))]
             (if (= eos next-token)
               acc
               (let [acc (rf acc next-token)]
                 (if (reduced? acc)
                   @acc
                   (recur acc (llama-update ctx next-token (raw/llama_get_kv_cache_token_count ctx) num-threads))))))))))))

(defn generate
  "Returns a seqable/reducible sequence of strings generated from ctx with prompt."
  ([ctx prompt]
   (generate ctx prompt nil))
  ([ctx prompt opts]
   (eduction
    (decode-token ctx)
    (generate-tokens ctx prompt opts))))


(defn generate-string
  "Returns a string with all tokens generated from prompt up until end of sentence or max context size."
  ([ctx prompt]
   (generate-string ctx prompt nil))
  ([ctx prompt opts]
   (let [[prompt-token-count _] (tokenize ctx prompt true)]
     (str/join
      (eduction
       (take (- (raw/llama_n_ctx ctx)
                prompt-token-count))
       (decode-token-to-char ctx)
       (generate-tokens ctx prompt opts))))))

(comment
  (def model-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin")
  (def model-path "../llama.cpp/models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q8_0.bin")
  (def model-path "../llama.cpp/models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q5_1.bin")
  (def model-path "models/llama2_7b_chat_uncensored.ggmlv3.q4_0.bin")
  (def model-path "models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q4_0.bin")

  (def ctx (create-context model-path {:n-gpu-layers 1
                                       :n-ctx 2048}))

  (require '[com.phronemophobic.llama.util.prompt :as prompt])
  (require '[com.phronemophobic.llama.util :as llutil])
  (llutil/print-response ctx "what is clojure?")

  (llutil/print-response ctx "The unicode emoji for :smile:"
                         {:seed 4321})


  ),

(defn -main [model-path prompt]
  (let [ctx (create-context model-path {:n-gpu-layers 1})
        [prompt-token-count _] (tokenize ctx prompt true)]
    (transduce
     (comp (take-while (fn [_]
                         (not (Thread/interrupted))))
           (take (- (raw/llama_n_ctx ctx)
                    prompt-token-count)))
     (completing
      (fn [_ s]
        (print s)
        (flush)))
     nil
     (generate ctx prompt))
    (println)))



