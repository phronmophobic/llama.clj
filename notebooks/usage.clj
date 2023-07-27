^{:nextjournal.clerk/visibility {:code :hide :result :hide}
  :nextjournal.clerk/toc true}
(ns usage
  (:require [nextjournal.clerk :as clerk]
            [nextjournal.clerk.viewer :as v]
            [util.viewers :refer [wrap-seed]]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [clojure.string :as str]))

{:nextjournal.clerk/visibility {:code :show :result :hide}}

;; # llama.clj

;; [llama.clj](https://github.com/phronmophobic/llama.clj) is a clojure wrapper for the [llama.cpp](https://github.com/ggerganov/llama.cpp) library.

;; ## Dependency

;; deps.edn dependency:

;; ```clojure
;; com.phronemophobic/llama-clj {:mvn/version "0.2"}
;; ```

;; ## Requires

;; All of the docs assume the following requires:

;; ```clojure
;; (require '[com.phronemophobic.llama :as llama])
;; ```

;; Throughout these docs, we'll be using the llama 7b chat model.
;; and the following context based on this model.
^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/code ";; downloaded previously
(def llama7b-path \"models/llama-2-7b-chat.ggmlv3.q4_0.bin\")
(def llama-context (llama/create-context llama7b-path {:n-gpu-layers 1}))
")

^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(do
  (def llama7b-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin")
  (def llama-context (llama/create-context llama7b-path {:n-gpu-layers 1}))
  (def seed 4321))

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
;; The `model-path` arg should be a string path (relative or absolute) to a F16, Q4_0, Q4_1, Q5_0, Q5_1, or Q8_0 ggml model.

;; ## Token Generation

;; Once a context is created, it can then be passed to `llama/generate-tokens`. The `llama/generate-tokens` function returns seqable or reducible sequence of tokens given a prompt. That means generated tokens can be processed using all of the normal clojure sequence and transducer based functions.

(wrap-seed
 (first (llama/generate-tokens llama-context "Hello World")))
(wrap-seed
 (clojure.string/join
   (eduction
    (map (fn [token] (raw/llama_token_to_str llama-context token)))
    (take 10)
    (llama/generate-tokens llama-context "Hello World"))))

;; ## Generating Text

;; Working with raw tokens is useful in some cases, but most of the time, it's more useful to work with a generated sequence of strings corresponding to those tokens. Lllama.clj provides a simple wrapper of `llama/generate-tokens` for that purpose, `llama/generate`.

(wrap-seed
 (into []
       (take 5)
       (llama/generate llama-context "Write a haiku about documentation.")))

;; If results don't need to be streamed, then `llama/generate-string` can be used to return a string with all the generated text up to the max context size.

(wrap-seed
 (llama/generate-string llama-context "Write a haiku about documentation."))

^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(comment
  (clerk/serve! {:watch-paths ["notebooks/usage.clj"]})
  
  (clerk/show! "notebooks/usage.clj")

  (clerk/build! {:paths ["notebooks/usage.clj"]
                 :out-path "docs/"
                 :bundle true})

  ,)
