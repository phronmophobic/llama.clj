(ns com.phronemophobic.llama
  (:require [com.phronemophobic.llama.raw :as raw]
            [clojure.string :as str])
  (:import java.lang.ref.Cleaner
           com.sun.jna.Memory
           com.sun.jna.Pointer
           com.sun.jna.ptr.IntByReference
           com.sun.jna.ptr.FloatByReference
           com.sun.jna.Structure))

(raw/import-structs!)
(defonce cleaner (delay (Cleaner/create)))

(defn eos []
  (raw/llama_token_eos))
(defn bos []
  (raw/llama_token_bos))

(def token-data-size (.size (llama_token_data.)))

(defn ^:private generate-tokens [ctx ^Memory token-buf num-tokens]
  (let [max-context-size (raw/llama_n_ctx ctx)]
    (loop [num-tokens num-tokens
           candidates-buf nil]
      (let [token-count (raw/llama_get_kv_cache_token_count ctx)]
        (if (< token-count max-context-size)
          (do
            (raw/llama_eval ctx token-buf num-tokens token-count 1)
            (let [n-vocab (raw/llama_n_vocab ctx)
                  logits (-> ^FloatByReference (raw/llama_get_logits ctx)
                             .getPointer
                             (.getFloatArray 0 n-vocab))

                  buf-size (* token-data-size n-vocab)
                  candidates-buf (if (and candidates-buf
                                          (>= (.size ^Memory candidates-buf)
                                              buf-size))
                                   candidates-buf
                                   (Memory. buf-size))]
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
                                  (.writeField "sorted" (byte 0)))

                    new-token-id (raw/llama_sample_token_greedy ctx candidates*)]
                ;; (prn new-token-id (llama_token_to_str ctx new-token-id))
                (print (raw/llama_token_to_str ctx new-token-id))
                (flush)
                (when (not= new-token-id
                            (raw/llama_token_eos))
                  (.setInt token-buf 0 new-token-id)
                  (recur 1
                         candidates-buf))))))))))



(defn ^:private llm-prompt [model-path prompt]
  (raw/llama_backend_init 0)
  (let [params (doto ^llama_context_params (raw/llama_context_default_params)
                 ;; (.writeField "n_gpu_layers" (int 1))
                 )
        model (raw/llama_load_model_from_file model-path params)
        _(assert model)
        context (raw/llama_new_context_with_model model params)
        ;; // Add a space in front of the first character to match OG llama tokenizer behavior
        prompt (str " " prompt)
        add-bos 1
        ;; tokens are ints
        num-tokens (* 4 (+ add-bos (count prompt)))
        token-buf (Memory. num-tokens)
        embd_inp (raw/llama_tokenize context prompt token-buf num-tokens add-bos)

        n-ctx (raw/llama_n_ctx context)


        _ (assert (< embd_inp (- n-ctx 4)) "prompt too long")

        ;; // do one empty run to warm up the model
        #_(let [tmp (IntByReference. (raw/llama_token_bos))]
            (raw/llama_eval context tmp 1 0 1))

        ;; (e arr (.toArray (llama_token_dataByReference.) 3) )


        max-context-size n-ctx]
    (generate-tokens context token-buf embd_inp)))


(defonce ^:private llm-init
  (delay
    (raw/llama_backend_init 0)))

(defn ^:private ->bool [b]
  (if b
    (byte 1)
    (byte 0)))

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
       :tensor-split (.writeField params "tensor_split" (float-array v))
       :rope-freq-base (.writeField params "rope_freq_base" (float v))
       :rope-freq-scale (.writeField params "rope_freq_scale" (float v))
       ;; :progress-callback (.writeField params "progress_callback" v)
       ;; :progress-callback-user-data (.writeField params "progress_callback_user_data" v)
       :low-vram (.writeField params "low_vram" (->bool v))
       :f16-kv (.writeField params "f16_kv" (->bool v))
       :logits-all (.writeField params "logits_all" (->bool v))
       :vocab-only (.writeField params "vocab_only" (->bool v))
       :use-mmap (.writeField params "use_mmap" (->bool v))
       :use-mlock (.writeField params "use_mlock" (->bool v))
       :embedding (.writeField params "embedding" (->bool v)))
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
  - `:f16-kv`: use fp16 for KV cache
  - `:logits-all`: the llama_eval() call computes all logits, not just the last one
  - `:vocab-only`: only load the vocabulary, no weights
  - `:use-mmap`: use mmap if possible
  - `:use-mlock`: force system to keep model in RAM
  - `:embedding`: embedding mode only
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
            embedding]
     :as params}]
   @llm-init
   (let [^llama_context_params
         llama-params (map->llama-params params)
         model (raw/llama_load_model_from_file model-path llama-params)
         _ (when (nil? model)
             (throw (ex-info "Error creating model"
                             {:params params
                              :model-path model-path})))
         context (raw/llama_new_context_with_model model llama-params)]

     ;; cleanup
     (let [ctx-ptr (Pointer/nativeValue context)
           model-ptr (Pointer/nativeValue model)
           model-ref (volatile! model)]
       (.register ^Cleaner @cleaner context
                  (fn []
                    (raw/llama_free ctx-ptr)

                    ;; make sure model doesn't lose
                    ;; all references and get garbage
                    ;; collected until context is freed.
                    (vreset! model-ref nil)))
       (.register ^Cleaner @cleaner model
                  (fn []
                    (raw/llama_free_model model-ptr))))

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
  ([ctx s]
   (llama-update ctx s (raw/llama_get_kv_cache_token_count ctx)))
  ([ctx s n-past]
   (let [[num-tokens token-buf]
         (cond
           (string? s)
           (tokenize ctx s (zero? n-past))

           (integer? s)
           (let [^Memory buf (get-token-buf ctx 1)]
             [1 (doto buf
                  (.setInt 0 s))]))]
     (raw/llama_eval ctx token-buf num-tokens n-past 1)
     ctx)))

(defn sample-logits-greedy [logits]
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

(defn get-logits [ctx]
  (let [n-vocab (raw/llama_n_vocab ctx)]
    (-> ^FloatByReference (raw/llama_get_logits ctx)
        .getPointer
        (.getFloatArray 0 n-vocab))))

(defn generate
  "Returns a seqable/reducible sequence of tokens from ctx from prompt."
  ([ctx prompt]
   (generate ctx prompt nil))
  ([ctx prompt {:keys [samplef
                       num-threads
                       seed
                       resize-context]
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
                      (lazy-seq (next (llama-update ctx next-token)))))))
          (llama-update ctx prompt 0)))
       clojure.lang.IReduceInit
       (reduce [_ rf init]
         (when seed
           (raw/llama_set_rng_seed ctx seed))
         (loop [acc init
                ret (llama-update ctx prompt 0)]
           (let [next-token (samplef (get-logits ctx))]
             (if (= eos next-token)
               acc
               (let [acc (rf acc next-token)]
                 (if (reduced? acc)
                   @acc
                   (recur acc (llama-update ctx next-token))))))))))))


(defn generate-response
  ([ctx prompt]
   (generate-response ctx prompt nil))
  ([ctx prompt opts]
   (let [[prompt-token-count _] (tokenize ctx prompt true)]
     (str/join
      (eduction
       (take (- (raw/llama_n_ctx ctx)
                prompt-token-count))
       (map #(raw/llama_token_to_str ctx %))
       (generate ctx prompt nil))))))

(comment
  (def model-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin")
  (def model-path "../llama.cpp/models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q8_0.bin")
  (def model-path "../llama.cpp/models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q5_1.bin")

  (def ctx (create-context model-path {:n-gpu-layers 1}))

  (def prompt "What is clojure?")
  ;; updates context logits
  (llama-update ctx prompt)

  (def results
    (loop [results [prompt]]
      (let [next-token (sample-logits-greedy (get-logits ctx))
            next-str (raw/llama_token_to_str ctx next-token)]
        (print next-str)
        (flush)
        (if (not= next-token
                  (raw/llama_token_eos))
          (do
            (llama-update ctx next-token)
            (recur (conj results next-str)))
          results))))

  ,)

(defn -main [model-path prompt]
  ;; "/Users/adrian/workspace/llama.cpp/models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q4_0.bin"
  (llm-prompt model-path prompt))
