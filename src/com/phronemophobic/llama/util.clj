(ns com.phronemophobic.llama.util
  (:require [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [clojure.string :as str])
  (:import com.sun.jna.Memory))

(defn next-token
  "Give a sequence of tokens, return the next token."
  [ctx tokens]
  (let [ctx (reduce (fn [ctx token]
                      (llama/llama-update ctx token))
                    (llama/llama-update ctx (first tokens) 0)
                    (rest tokens))]
    (llama/sample-logits-greedy (llama/get-logits ctx))))

(defn tokenize
  "Tokenize the string s into a collection of int tokens."
  [ctx s]
  (let [;; tokens are int32s
        buf-size (* 4 (count s))
        token-buf (Memory. buf-size)
        num-tokens (raw/llama_tokenize ctx s token-buf (count s) 0)]
    (vec (.getIntArray token-buf 0 num-tokens))))

(defn untokenize
  "Given a sequence of tokens, return the string representation."
  [ctx tokens]
  (str/join (map #(raw/llama_token_to_str ctx %) tokens)))

(defn print-response
  "Generates a response from prompt and print the results as they become available.

  Returns nil"
  [ctx prompt opts]
  (transduce
   (comp (map #(raw/llama_token_to_str ctx %))
         (take-while (fn [_]
                       (not (Thread/interrupted)))))
   (completing
    (fn [_ s]
      (print s)
      (flush)))
   nil
   (llama/generate ctx prompt opts)))
