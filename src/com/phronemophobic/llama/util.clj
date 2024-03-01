(ns com.phronemophobic.llama.util
  (:require [com.phronemophobic.llama :as llama]
            ;; [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.impl.model :as model]
            [clojure.string :as str])
  (:import com.sun.jna.Memory))

(defn next-token
  "Give a sequence of tokens, return the next token."
  [ctx tokens]
  (let [ctx (reduce (fn [ctx token]
                      (llama/llama-update ctx token))
                    (llama/llama-update ctx (first tokens) 0)
                    (rest tokens))]
    (llama/sample-logits-greedy (model/get-logits ctx))))

(defn tokenize
  "Tokenize the string s into a collection of int tokens."
  [ctx s]
  (let [[num-tokens token-buf]
        (model/tokenize ctx s false)]
    (vec (.getIntArray token-buf 0 num-tokens))))

(defn untokenize
  "Given a sequence of tokens, return the string representation."
  [ctx tokens]
  (str/join
   (eduction
    (model/decode-token-to-str ctx)
    tokens)))

(defn print-response
  "Generates a response from prompt and print the results as they become available.

  Returns nil"
  ([ctx prompt]
   (print-response ctx prompt nil))
  ([ctx prompt opts]
   (transduce
    (take-while (fn [_]
                  (not (Thread/interrupted))))
    (completing
     (fn [_ s]
       (print s)
       (flush)))
    nil
    (llama/generate ctx prompt opts))))

(defn normalize-embedding
  "Normalize the embedding `emb` so that it matches output from llama.cpp's ./embedding example."
  [emb]
  (let [n (alength emb)
        norm (loop [norm (float 0)
                    i 0]
               (if (< i n)
                 (let [x (aget emb i)]
                   (recur (+ norm (* x x))
                          (inc i)))
                 norm))
        norm (Math/sqrt norm)]
    (float-array
     (eduction
      (map (fn [i]
             (/ (aget emb i)
                norm)))
      (range n)))))
