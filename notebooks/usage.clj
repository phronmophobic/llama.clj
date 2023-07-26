^{:nextjournal.clerk/visibility {:code :hide :result :hide}
  :nextjournal.clerk/toc true}
(ns usage
  (:require [nextjournal.clerk :as clerk]
            [nextjournal.clerk.viewer :as v]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [clojure.string :as str])
  (:import com.sun.jna.Memory)
  )

{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(def graph-viewer
  {:pred ::generate
   :transform-fn
   (clerk/update-val
    (fn [m]
      
      (let [blocks []
            blocks (if-let [code (::code m)]
                     (conj blocks
                           (clerk/md
                            (str "```clojure\n"

                                 (with-out-str
                                   (clojure.pprint/pprint
                                    (::code m)))
                                 "\n```")))
                     blocks)]
        
        
        
        (apply
         clerk/col
         {::clerk/width :wide}
         blocks))))})

(clerk/add-viewers! [graph-viewer])


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

(defonce eval-memo (memoize eval))
(def llama-viewer
  {:pred ::generate
   :transform-fn
   (clerk/update-val
    (fn [m]

      (raw/llama_set_rng_seed llama-context
                              (get m ::seed seed))
      (let [blocks []
            code (::generate m)
            result (eval-memo code)
            blocks (conj blocks
                         (clerk/md
                          (str "```clojure\n"

                               (with-out-str
                                 (clojure.pprint/pprint
                                  (::generate m)))
                               "\n```")))

            blocks (conj blocks
                         (clerk/md
                          (str "```clojure\n"
                               result
                               "\n```")))]
        
        (apply
         clerk/col
         {::clerk/width :wide}
         blocks))))})

(clerk/add-viewers! [llama-viewer])

{:nextjournal.clerk/visibility {:code :hide :result :show}}


;; ## Overview

;; The llama.clj API built on top of two functions, `llama/create-context` and `llama/generate-tokens`. The `create-context` builds a context that can be used (and reused) to generate tokens. Let's try a few examples.

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

{::generate '(first (llama/generate-tokens llama-context "Hello World"))}
{::generate '(clojure.string/join
              (eduction
               (map #(raw/llama_token_to_str llama-context %))
               (take 10)
               (llama/generate-tokens llama-context "Hello World")))}

;; ## Generating Text

;; Working with raw tokens is useful in some cases, but most of the time, it's more useful to work with a generated sequence of strings corresponding to those tokens. Lllama.clj provides a simple wrapper of `llama/generate-tokens` for that purpose, `llama/generate`.

{::generate '(clojure.string/join
              (llama/generate llama-context "Write a haiku about documentation."))}










^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(comment
  (clerk/serve! {:watch-paths ["notebooks/usage.clj"]})
  
  (clerk/show! "notebooks/usage.clj")

  (clerk/build! {:paths ["notebooks/usage.clj"]
                 :out-path "docs/"
                 :bundle true})

  ,)
