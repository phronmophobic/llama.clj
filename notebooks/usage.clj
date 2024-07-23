^{:nextjournal.clerk/visibility {:code :hide :result :hide}
  :nextjournal.clerk/toc true}
(ns usage
  (:require [nextjournal.clerk :as clerk]
            [nextjournal.clerk.viewer :as v]
            [util.viewers :refer [wrap-seed]]
            [com.phronemophobic.llama :as llama]
            ;; required to make clerk work.
            [com.phronemophobic.llama.raw-gguf :as raw]
            [clojure.string :as str]))

{:nextjournal.clerk/visibility {:code :show :result :hide}}

;; # llama.clj

;; [llama.clj](https://github.com/phronmophobic/llama.clj) is a clojure wrapper for the [llama.cpp](https://github.com/ggerganov/llama.cpp) library.

;; ## Dependency

;; deps.edn dependency:

;; ```clojure
;; com.phronemophobic/llama-clj {:mvn/version "0.8.5"}
;; ```

;; ## Requires

;; All of the docs assume the following requires:

;; ```clojure
;; (require '[com.phronemophobic.llama :as llama])
;; ```

;; Throughout these docs, we'll be using the qwen 0.5b instruct model.
;; and the following context based on this model.
^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/code ";; downloaded previously from
;; https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_k_m.gguf?download=true
(def model-path \"models/qwen2-0_5b-instruct-q4_k_m.gguf\")
;; Use larger context size of 2048.
(def llama-context (llama/create-context model-path {:n-ctx 2048}))
")

^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(do
  (def model-path "models/qwen2-0_5b-instruct-q4_k_m.gguf")
  (def llama-context (llama/create-context model-path {:n-ctx 2048}))
  (def seed 1337)
  seed)


{:nextjournal.clerk/visibility {:code :hide :result :hide}}

(clerk/add-viewers! [util.viewers/llama-viewer])

{:nextjournal.clerk/visibility {:code :hide :result :show}}


;; ## Overview

;; The llama.clj API is built around two functions, `llama/create-context` and `llama/generate-tokens`. The `create-context` builds a context that can be used (and reused) to generate tokens.

;; ## Context Creation

;; The `llama/create-context` has two arities:
;; ```clojure
;; (llama/create-context model-path)
;; (llama/create-context model-path opts)
;; ```
;; If no `opts` are specified, then defaults will be used.
;; 
;; The `model-path` arg should be a string path (relative or absolute) to a gguf or ggml model.

;; ### Context Size

;; The default context size is 512 tokens, which can be limiting. To increase the context size, provide `:n-ctx` as an option during context creation.

;; ```clojure
;; ;; Use context size of 2048 tokens
;; (llama/create-context model-path {:n-ctx 2048})
;; ```

;; The max context size of the model can be used by passing `0` for `:n-ctx`.

;; ```clojure
;; ;; Use model's max context size.
;; (llama/create-context model-path {:n-ctx 0})
;; ```

;; ## Prompt Templates

;; Most chat or instruct models expect a specific prompt format. `llama.cpp` provides limited support for applying chat templates. The `chat-apply-template` offers templates for many popular models and formats. Some less popular models may require custom templating and is not included.

;; ### Model Provided Templates

;; Many newer gguf models include the prompt format they expect in their metadata:

{:nextjournal.clerk/visibility {:code :show :result :show}}
(get (llama/metadata llama-context) "tokenizer.chat_template")

;; If the template is included and llama.cpp recognizes it, then the template can be applied using `llama/chat-apply-template`.

(llama/chat-apply-template llama-context
                           [{:role "user"
                             :content "What's the best way to code in clojure?"}])

;; Typical roles are \"assistant\", \"system\", and \"user\". It is best to check the documentation for your particular model to see which roles are available. Also note that llama.cpp's template detection isn't exact and may guess incorrectly in some cases.

;; ### Applying Templates By Name

;; Even if a model doesn't include a particular template, many models use one of the popular template formats. In those cases, you can pass in a template name.

(llama/chat-apply-template "llama3"
                           [{:role "user"
                             :content "What's the best way to code in clojure?"}])

;; See the doc string of `chat-apply-template` for a list of allowed template names.

{:nextjournal.clerk/visibility {:code :hide :result :show}}

;; ## Token Generation

;; Once a context is created, it can then be passed to `llama/generate-tokens`. The `llama/generate-tokens` function returns seqable or reducible sequence of tokens given a prompt. That means generated tokens can be processed using all of the normal clojure sequence and transducer based functions.


^{:nextjournal.clerk/visibility {:code :show :result :show}}
(def hello-world-prompt
  (llama/chat-apply-template llama-context
                             [{:role "user"
                               :content "Hello World"}]))

(wrap-seed
 (first (llama/generate-tokens llama-context
                               hello-world-prompt)))
(wrap-seed
 (clojure.string/join
   (eduction
    (llama/decode-token llama-context)
    (take 10)
    (llama/generate-tokens llama-context hello-world-prompt))))

;; ## Generating Text

;; Working with raw tokens is useful in some cases, but most of the time, it's more useful to work with a generated sequence of strings corresponding to those tokens. Lllama.clj provides a simple wrapper of `llama/generate-tokens` for that purpose, `llama/generate`.

^{:nextjournal.clerk/visibility {:code :show :result :show}}
(def haiku-prompt
  (llama/chat-apply-template
   llama-context
   [{:role "user"
     :content "Write a short poem about documentation."}]))

(wrap-seed
 (into []
       (take 5)
       (llama/generate llama-context
                       haiku-prompt)))

;; If results don't need to be streamed, then `llama/generate-string` can be used to return a string with all the generated text up to the max context size.

(wrap-seed
 (llama/generate-string llama-context haiku-prompt))
;; ## Log Callback

;; By default, llama.cpp's logs are sent to stderr (note: stderr is different from `*err*` and `System/err`).
;; The log output can be redirected by setting a log callback.

;; ```clojure
;; ;; disable logging
;; (llama/set-log-callback (fn [& args]))
;; ;; print to stdout
;; (llama/set-log-callback
;;  (fn [log-level msg]
;;    (let [level-str (case log-level
;;                      2 "error"
;;                      3 "warn"
;;                      4 "info"
;;                      5 "debug")]
;;      (println log-level msg))))
;; ```

;; ## Generating Embeddings

{:nextjournal.clerk/visibility {:code :show :result :show}}

;; To generate embeddings, contexts must be created with `:embedding` set to `true`.
(def llama-embedding-context
  (llama/create-context model-path
                        {:embedding true}))




(vec
 (llama/generate-embedding llama-embedding-context "hello world"))


;; ## FAQ

;; ### Context size exceeded

;; This exception means that the maximum number of tokens for a particular context have been generated and that no more tokens can be generated. There are many options for handling generation beyond the context size that are beyond the scope of this documentation. However, one easy option is to increase the context size of the context if the context size is not already at its maximum (see [:n-ctx](#context-size)). The maximum context size will depend on your hardware and the model. However, there are tradeoffs to larger context sizes that can be mitigated with other techniques. The [Local LLama](https://www.reddit.com/r/LocalLLaMA) subreddit can be a good resource for practical tips.

^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(comment
  (clerk/serve! {:watch-paths ["notebooks/usage.clj"]})
  
  (clerk/show! "notebooks/usage.clj")

  (clerk/build! {:paths ["notebooks/usage.clj"]
                 :out-path "docs/"
                 :bundle true})

  ,)
