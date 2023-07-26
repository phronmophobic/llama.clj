^{:nextjournal.clerk/visibility {:code :hide :result :hide}
  :nextjournal.clerk/toc true}
(ns docs
  (:require [nextjournal.clerk :as clerk]
            [nextjournal.clerk.viewer :as v]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [clojure.string :as str])
  (:import com.sun.jna.Memory)
  )

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
;; (require '[com.phronemophobic.llama.util :as llutil])
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

;; ## Quick Start

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/code
   '(llama/generate-response llama-context "What are some reasons for learning clojure?"))

^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(comment


  #_reseed ^{:nextjournal.clerk/visibility {:code :hide :result :hide} ::clerk/no-cache true} (def llama-context (doto llama-context (raw/llama_set_rng_seed seed)))

  (def tokens
    (llama/generate llama-context "What are some reasons for learning clojure?")))

;; _Note: the initial german word and the begining of the response is an artifact of how the llama2 chat model works_

#_reseed ^{:nextjournal.clerk/visibility {:code :hide :result :hide} ::clerk/no-cache true} (def llama-context (doto llama-context (raw/llama_set_rng_seed seed)))

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/md
 (llama/generate-response llama-context "What are some reasons for learning clojure?"))



;; ## Token Generation

;; ### Token Basics

;; One of main operations of an llm is to predict the next token
;; given a sequence of tokens.

{:nextjournal.clerk/visibility {:code :show :result :show}}
(llutil/next-token llama-context [10994 2787])


;; As you can see tokens are just ints and aren't actually that useful in their raw form.
;; Fortunately, it's easy to generate tokens from text and vice versa.
(def hello-tokens (llutil/tokenize llama-context "Hello World"))
(llutil/untokenize llama-context hello-tokens)

;; _Note: the method for tokenizing text is a large topic by itself
;; and is beyond the scope of this guide._

;; Now that we know how to tokenize and untokenize, let's revisit
;; predicting the next token of "Hello World".

(def hello-next-token
  (llutil/next-token llama-context
                     (llutil/tokenize llama-context "Hello World")))
(def hello-next-text (llutil/untokenize llama-context [hello-next-token]))

;; The next predicted token for "Hello World" is `"!"`!

;; ### Predicting multiple tokens

;; Predicting a single token isn't very useful.
;; Most of the time, we'll be interested in generating a "full"
;; response.

;; In general, llms only predict one token at a time, so how do
;; we get a full response? Easy, just add the predicted token
;; to our initial sequence of tokens and repeat until we're done.

;; How do we know when the llm is done? Helpfully, there is a
;; specific token for signifying that the llm is done
;; called "end of sentence" or `eos` for short. Since
;; we've mentioned `eos`, we should also mention the
;; `bos` token (beginning of sentence token). While it doesn't
;; matter for some models, it's good practice to start
;; token sequences with the bos token. 

;; Here's some example code for generating a full response from an llm:

^{:nextjournal.clerk/visibility {:code :show :result :hide}}
(def input-str "Write a haiku about coffee.")
(def input-tokens (into [;; don't forget the beginning of sentence token
                         (llama/bos)]
                        (llutil/tokenize llama-context input-str)))

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/code
 "(def full-response-tokens
  (loop [tokens input-tokens]
    (let [next-token (llutil/next-token llama-context tokens)]
      (if (= next-token (llama/bos))
        ;; we're done
        tokens
        (recur (conj tokens next-token))))))")

#_reseed ^{:nextjournal.clerk/visibility {:code :hide :result :hide} ::clerk/no-cache true} (def llama-context (doto llama-context (raw/llama_set_rng_seed seed)))

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(def full-response-tokens
  (into []
        (take 100)
        (llama/generate llama-context input-str)))

(llutil/untokenize llama-context full-response-tokens)

;; _Note: the initial german word and the begining of the response is an artifact of how the llama2 chat model works_

#_reseed ^{:nextjournal.clerk/visibility {:code :hide :result :hide} ::clerk/no-cache true} (def llama-context (doto llama-context (raw/llama_set_rng_seed seed)))

;; If you do run this snippet, you'll probably find that it's _very_ slow, even on a gpu. This naive implementation is easy to follow, but is not very efficient. Since generating a full response is a common use case, an optimized version that produces the same result is available.

^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(comment
  (into []
        (take 100)
        (llama/generate llama-context input-str)))
^{:nextjournal.clerk/visibility {:code :hide :result :show}}

(clerk/code '(llama/generate-response llama-context input-str))
#_reseed ^{:nextjournal.clerk/visibility {:code :hide :result :hide} ::clerk/no-cache true} (def llama-context (doto llama-context (raw/llama_set_rng_seed seed)))
^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/md
 (str
  "```\n"
  (llama/generate-response llama-context input-str)
  "\n```"))



^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(comment
  (clerk/serve! {:watch-paths ["notebooks/docs.clj"]})
  
  (clerk/show! "notebooks/docs.clj")

  (clerk/build! {:paths ["notebooks/docs.clj"]
                 :out-path "docs/"
                 :bundle true})

  (render-graph {:nodes ["A" "B"]}
                {:filename "tmp/graph.png"})
  
  ,)
