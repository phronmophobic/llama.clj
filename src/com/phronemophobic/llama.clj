(ns com.phronemophobic.llama
  (:require
   ;; [com.phronemophobic.llama.raw :as raw]
   [com.phronemophobic.llama.impl.model :as model]
            [clojure.string :as str])
  (:import java.nio.charset.CodingErrorAction
           java.nio.charset.CharsetDecoder
           java.nio.charset.Charset
           java.nio.ByteBuffer
           java.nio.CharBuffer
           com.sun.jna.Memory
           com.sun.jna.Pointer
           com.sun.jna.ptr.IntByReference
           com.sun.jna.ptr.FloatByReference
           com.sun.jna.Structure))

(def ^:dynamic
  *num-threads*
  "Number of threads used when generating tokens."
  (.. Runtime getRuntime availableProcessors))

;; (def ^:private token-data-size (.size (llama_token_data.)))

(defn eos
  "Returns the llama end of sentence token.

  Calling `eos` without a context is deprecated as not all models use the same bos token."
  ;; only for backwards compatibility
  ([]
   (int 2))
  ([ctx]
   (model/token-eos ctx)))
(defn bos
  "Returns the llama beginning of sentence token.

  Calling `bos` without a context is deprecated as not all models use the same bos token."
  ;; only for backwards compatibility
  ([]
   (int 1))
  ([ctx]
   (model/token-bos ctx)))

(defn end-of-generation?
  "Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)"
  [ctx token]
  (model/token-is-eog ctx token))

(defn metadata
  "Returns a map of the metadata associated with ctx."
  [ctx]
  (model/metadata ctx))

(defn model-description
  "Get a string describing the model type."
  [ctx]
  (model/model-description ctx))

(defn model-size
  "Returns the total size of all the tensors in the model in bytes."
  [ctx]
  (model/model-size ctx))

(defn model-n-params
  "Returns the total number of parameters in the model."
  [ctx]
  (model/model-n-params ctx))

(defn n-vocab
  "The number of available tokens for the associated model."
  [ctx]
  (model/n-vocab ctx))

(defn n-embd
  "The length of the embedding vector for the associated model."
  [ctx]
  (model/n-embd ctx))

(defn n-ctx
  "The context size for the associated model."
  [ctx]
  (model/n-ctx ctx))

(def ^:private ggml-model
  (delay
    @(requiring-resolve 'com.phronemophobic.llama.raw/llama-model)))
(def ^:private gguf-model
  (delay
    @(requiring-resolve 'com.phronemophobic.llama.raw-gguf/llama-model)))

(defonce ^:private log-callback (atom nil))
(defn set-log-callback
  "Sets the log callback. The callback should be a function that recieves two args: log level and msg.
  Setting to nil will cause output to be written to stderr.
  The log callback is global for all contexts.

  The log levels are as follows:
        GGML_LOG_LEVEL_ERROR = 2,
        GGML_LOG_LEVEL_WARN  = 3,
        GGML_LOG_LEVEL_INFO  = 4,
        GGML_LOG_LEVEL_DEBUG = 5

  Only supported for gguf models.

  Example:
  (set-log-callback ctx (fn [level msg]
                          (println level msg)))"
  
  [cb]
  (let [[old new] (reset-vals! log-callback
                               (fn [level msg _user-info]
                                 (cb level (.getString (.getPointer msg) 0 "utf-8"))))]
    (model/set-log-callback @gguf-model new)
   ;; try to hang onto old reference until new is set
    (identity old)))

(defn chat-apply-template
  "Returns a string with chat `messages` formatted using the format associated with `ctx`.

  Args:
  `template`: A llama context or a template name. Templates names
  are one of:
  `#{\"chatml\", \"llama2\", \"phi3\", \"zephyr\", \"monarch\",
  \"gemma\", \"orion\", \"openchat\", \"vicuna\",
  \"deepseek\", \"command-r\", \"llama3\"}`

  `messages`: a sequence of chat messages. chat messages are maps with `:role` and `:content`.
  Typical roles are \"assistant\", \"system\", and \"user\".

  `opts`: A map with the following options:
    `:append-start-assistant-message?`: Whether to end the prompt with the token(s) that
                                        indicate the start of an assistant message.
                                        If omitted, defaults to true.

  Throws `IllegalArgumentException` if the template format is unsupported.
  See: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template

  Throws `UnsupportedOperationException` for ggml models.

  Example:
  (chat-apply-template
   ctx
   [{:role \"assistant\" :content \"You are a friendly, helpful assistant.\"}
    {:role \"user\" :content \"What is clojure?\"}]
   true)
  "
  ([template messages]
   (chat-apply-template template messages {}))
  ([template messages opts]
   (model/chat-apply-template @gguf-model template messages opts)))

(defn create-context
  "Create and return an opaque llama context.

  `model-path` should be an absolute or relative path to a ggml or gguf model.

  An optional map of parameters may be passed for parameterizing the model. The following keys map to their corresponding llama.cpp equivalents:
  - `:seed`: RNG seed, -1 for random
  - `:n-ctx`: context size, set to 0 to use context size from model
  - `:n-batch`: prompt processing batch size
  - `:n-gpu-layers`: number of layers to store in VRAM
  - `:main-gpu`: the GPU that is used for scratch and small tensors
  - `:tensor-split`: how to split layers across multiple GPUs
  - `:rope-freq-base`: RoPE base frequency
  - `:rope-freq-scale`: RoPE frequency scaling factor
  - `:low-vram`: if true, reduce VRAM usage at the cost of performance (ggml only)
  - `:mul_mat_q`: if true, use experimental mul_mat_q kernels
  - `:f16-kv`: use fp16 for KV cache
  - `:logits-all`: the llama_eval() call computes all logits, not just the last one
  - `:vocab-only`: only load the vocabulary, no weights
  - `:use-mmap`: use mmap if possible
  - `:use-mlock`: force system to keep model in RAM
  - `:embedding`: if true, extract embeddings (together with logits)
  - `:gqa`: grouped-query attention factor (TEMP!!! use 8 for LLaMAv2 70B) (ggml only)
  - `:rms-norm-eps`: rms norm eps (TEMP!!! use 1e-5 for LLaMAv2) (ggml only)

  The `:model-format` can be specified as either `:ggml` or `:gguf`. If not provided,
  the model format will be guessed by looking at `model-path`.

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
            mul_mat_q
            f16-kv
            logits-all
            vocab-only
            use-mmap
            use-mlock
            embedding
            gqa
            rms-norm-eps
            model-format]
     :as params}]
   (let [format
         (cond
           model-format model-format
           (str/ends-with? model-path ".ggml") :ggml
           (str/ends-with? model-path ".gguf") :gguf
           (str/includes? model-path "ggml") :ggml
           :else :gguf)
         libllama
         (case format
           :ggml @ggml-model
           :gguf @gguf-model)]
     (model/create-context
      libllama
      model-path
      params))))


(defn get-logits
  "Returns a copy of the current context's logits as a float array."
  [ctx]
  (model/get-logits ctx))

(defn get-embedding
  "Returns a copy of the current context's embedding as a float array.

  The context should have been created with the `:embedding` option set to true."
  [ctx]
  (model/get-embedding ctx))

(defn set-rng-seed
  "Manually set the rng seed for a context."
  [ctx seed]
  (model/set-rng-seed ctx seed))

(defn llama-update
  "Adds `s` to the current context and updates the context's logits (see `get-logits`).

  `s`: either be a string or an integer token.
  `n-past`: number of previous tokens to include when updating logits.
  `num-threads`: number of threads to use when updating the logits.
                 If not provided, or `nil`, defaults to `*num-threads*`.
  "
  ([ctx s]
   (model/eval ctx s))
  ([ctx s n-past]
   (model/eval ctx s n-past *num-threads*))
  ([ctx s n-past num-threads]
   (let [num-threads (or num-threads *num-threads*)]
     (model/eval ctx s n-past *num-threads*))))

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
             [nil Float/NEGATIVE_INFINITY]
             logits))



(defn init-mirostat-v2-sampler
  "Given a context, returns a sampling function that uses the llama.cpp mirostat_v2 implementation."
  ([ctx]
   (let [tau (float 5.0)
         eta (float 0.1)]
     (init-mirostat-v2-sampler ctx tau eta)))
  ([ctx tau eta]
   (fn [logits]
     (model/sample-mirostat-v2 ctx
                               (volatile! nil)
                               (volatile! (* 2 tau))
                               tau
                               eta))))




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

(defn decode-token-to-char
  "Returns a transducer that expects a stream of llama tokens
  and outputs a stream of decoded chars.

  The transducer will buffer intermediate results until enough
  bytes to decode a character are available."
  ([ctx]
   (model/decode-token-to-char ctx nil))
  ([ctx opts]
   (model/decode-token-to-char ctx opts)))

(defn decode-token
  "Returns a transducer that expects a stream of llama tokens
  and outputs a stream of strings.

  The transducer will buffer intermediate results until enough
  bytes to decode a character are available. Also combines
  surrogate pairs of characters."
  ([ctx]
   (model/decode-token-to-str ctx nil))
  ([ctx opts]
   (model/decode-token-to-str ctx opts)))

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
                     (init-mirostat-v2-sampler ctx))]
     (reify
       clojure.lang.Seqable
       (seq [_]
         (when seed
           (model/set-rng-seed ctx seed))
         ((fn next [ctx]
            (let [next-token (samplef (model/get-logits ctx))]
              (when (not (end-of-generation? ctx next-token))
                (cons next-token
                      (lazy-seq (next (model/eval ctx next-token nil num-threads)))))))
          (llama-update ctx prompt 0 num-threads)))
       clojure.lang.IReduceInit
       (reduce [_ rf init]
         (when seed
           (model/set-rng-seed ctx seed))
         (loop [acc init
                ret (llama-update ctx prompt 0 num-threads)]
           (let [next-token (samplef (model/get-logits ctx))]
             (if (end-of-generation? ctx next-token)
                 acc
               (let [acc (rf acc next-token)]
                 (if (reduced? acc)
                   @acc
                   (recur acc (llama-update ctx next-token nil num-threads))))))))))))

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
   (let [[prompt-token-count _] (model/tokenize ctx prompt true)]
     (str/join
      (eduction
       (take (- (model/n-ctx ctx)
                prompt-token-count))
       (model/decode-token-to-str ctx)
       (generate-tokens ctx prompt opts))))))


(defn generate-embedding
  "Returns the embedding for a given input prompt.

  The context should have been created with the `:embedding` option set to true.

  Note: embeddings are not normalized. See `com.phronemophobic.llama.util/normalize-embedding.`"
  ([ctx prompt opts]
   (llama-update ctx prompt 0 (:num-threads opts))
   (get-embedding ctx))
  ([ctx prompt]
   (llama-update ctx prompt 0 *num-threads*)
   (get-embedding ctx)))

(comment
  (def model-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin")
  (def model-path "../llama.cpp/models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q8_0.bin")
  (def model-path "../llama.cpp/models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q5_1.bin")
  (def model-path "models/llama2_7b_chat_uncensored.Q4_0.gguf")
  (def model-path "models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q4_0.bin")

  ;; https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_0.gguf
  (def model-path "models/mistral-7b-instruct-v0.1.Q4_0.gguf")
  (def model-path "models/llama-2-7b-chat.Q4_0.gguf")
  (def model-path "models/qwen2-0_5b-instruct-q4_0.gguf")

  (def model-path "models/bge-large-en-v1.5-q4_k_m.gguf")

  (def ctx (create-context model-path {:n-ctx 0
                                       ;; :embedding true
                                       ;;:n-gpu-layers 1
                                       }))

  (require '[com.phronemophobic.llama.util.prompt :as prompt])
  (require '[com.phronemophobic.llama.util :as llutil])
  (llutil/print-response ctx "what is clojure?")

  (generate-string
   ctx
   (chat-apply-template ctx
                        [{:role "user"
                          :content "what is clojure?"}]))


  ),

(defn -main [model-path prompt]
  (let [ctx (create-context model-path)
        formatted-prompt (try
                           (chat-apply-template ctx
                                                [{:role "user"
                                                  :content prompt}])
                           (catch IllegalArgumentException e
                             prompt)
                           (catch UnsupportedOperationException e
                             prompt))]
    (transduce
       (take-while (fn [_]
                     (not (Thread/interrupted))))
       (completing
        (fn [_ s]
          (print s)
          (flush)))
       nil
       (generate ctx formatted-prompt))
    (println)))



