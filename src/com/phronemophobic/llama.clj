(ns com.phronemophobic.llama
  (:require
   ;; [com.phronemophobic.llama.raw :as raw]
   [com.phronemophobic.llama.impl.model :as model]
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

;; (raw/import-structs!)
(defonce cleaner (delay (Cleaner/create)))

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

(def ^:private ggml-model
  (delay
    @(requiring-resolve 'com.phronemophobic.llama.raw/llama-model)))
(def ^:private gguf-model
  (delay
    @(requiring-resolve 'com.phronemophobic.llama.raw-gguf/llama-model)))

(defn create-context
  "Create and return an opaque llama context.

  `model-path` should be an absolute or relative path to a ggml or gguf model.

  An optional map of parameters may be passed for parameterizing the model. The following keys map to their corresponding llama.cpp equivalents:
  - `:seed`: RNG seed, -1 for random
  - `:n-ctx`: text context
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
  - `:embedding`: embedding mode only
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
                     (init-mirostat-v2-sampler ctx))
         eos (model/token-eos ctx)]
     (reify
       clojure.lang.Seqable
       (seq [_]
         (when seed
           (model/set-rng-seed ctx seed))
         ((fn next [ctx]
            (let [next-token (samplef (model/get-logits ctx))]
              (when (not= eos next-token)
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
             (if (= eos next-token)
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

(comment
  (def model-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin")
  (def model-path "../llama.cpp/models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q8_0.bin")
  (def model-path "../llama.cpp/models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q5_1.bin")
  (def model-path "models/llama2_7b_chat_uncensored.ggmlv3.q4_0.bin")
  (def model-path "models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q4_0.bin")

  ;; https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_0.gguf
  (def model-path "models/mistral-7b-instruct-v0.1.Q4_0.gguf")
  (def model-path "models/llama-2-7b-chat.Q4_0.gguf")

  (def ctx (create-context model-path {:n-ctx 2048
                                       ;;:n-gpu-layers 1
                                       }))

  (require '[com.phronemophobic.llama.util.prompt :as prompt])
  (require '[com.phronemophobic.llama.util :as llutil])
  (llutil/print-response ctx "what is clojure?")

  (llutil/print-response ctx "what is clojure")


  ),

(defn -main [model-path prompt]
  (let [ctx (create-context model-path)]
    (transduce
     (take-while (fn [_]
                   (not (Thread/interrupted))))
     (completing
      (fn [_ s]
        (print s)
        (flush)))
     nil
     (generate ctx prompt))
    (println)))



