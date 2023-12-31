(ns com.phronemophobic.llama.token-test
  (:require [clojure.test :refer :all]
            [clojure.string :as str]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.spec.alpha :as s]
            [clojure.test.check :as tc]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.impl.model :as model]
            [com.phronemophobic.llama.util :as llutil]
            [com.phronemophobic.llama.raw :as raw])
  (:import com.sun.jna.Pointer))


 (defn tokens->bytes [ps]
   (loop [bs []
          idx 0
          offset 0]
     (if-let [p (nth ps idx nil)]
       (let [b (.getByte p offset)]
         (if (zero? b)
           (recur bs (inc idx) 0)
           (recur (conj bs b) idx (inc offset))))
       bs)))

 (defn token->pointer [ctx tok]
   (.invoke 
    (.getFunction raw/libllama
                  "llama_token_to_str")
    Pointer (to-array [ctx (int tok)])))

(defn untokenize [ctx tokens]
  (String. (byte-array
            (tokens->bytes
             (mapv #(token->pointer ctx %) tokens))) "utf-8"))

(defn token-gen [ctx sgen]
  (let [gen (gen/such-that
                  (fn [s]
                    (and
                     (seq s)
                     (< (count s)
                        2048)
                     (not (str/includes? s " "))
                     #_(not (str/includes? s ""))))
                  sgen
                  1000)
        gen (gen/fmap
             (fn [s]
               (try
                 [s (llutil/tokenize ctx s)]
                 (catch AssertionError e
                   nil)))
             gen)]
    (gen/such-that some?
                   gen
                   {:max-tries 100})))


(defn check-tokens
  ([ctx s]
   (check-tokens
    ctx
    s
    (llutil/tokenize ctx s)))
  ([ctx s tokens]
   (= s
      (str/join
       (eduction
        (llama/decode-token-to-char ctx)
        tokens))))
  )

(defn token-roundtrip [ctx]
  (prop/for-all [[s tokens] (token-gen ctx gen/string-alphanumeric)]
                (= s
                   (str/join
                    (eduction
                     (llama/decode-token-to-char ctx)
                     tokens)))))


(defn token-roundtrip2 [ctx]
  (prop/for-all [[s tokens] (token-gen ctx gen/string)]
                (let [s2 (str/join
                          (eduction
                           (llama/decode-token-to-char ctx)
                           tokens))]
                  (= s s2))))


(defn token-roundtrip3 [ctx]
  (prop/for-all [[s tokens] (token-gen ctx gen/string-alphanumeric)]
                (let [s2 (untokenize ctx tokens)]
                  (= s s2))))


(defn tokenize-matches [ctx]
  (prop/for-all [[s tokens] (token-gen ctx gen/string)]
                (let [[num-tokens p] (model/tokenize ctx s false)]
                 (= tokens
                    (vec (.getIntArray p 0 num-tokens))))))

(comment
  (def model-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin")
  (def ctx (llama/create-context model-path {:n-gpu-layers 1}))

  (tc/quick-check 10000 (token-roundtrip ctx))
  (tc/quick-check 10000 (token-roundtrip2 ctx))

  (tc/quick-check 10000 (token-roundtrip3 ctx))

  (tc/quick-check 10000 (tokenize-matches ctx))

  (llutil/tokenize ctx "😊")
  ,)

(defspec token-roundtrip-spec
  10000
  (let [model-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin"]
    (let [ctx (llama/create-context model-path {:n-gpu-layers 1})]
      (token-roundtrip ctx))))


(defspec token-roundtrip2-spec
  10000
  (let [model-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin"]
    (let [ctx (llama/create-context model-path {:n-gpu-layers 1})]
      (token-roundtrip2 ctx))))


(defspec tokenize-matches-spec
  10000
  (let [model-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin"]
    (let [ctx (llama/create-context model-path {:n-gpu-layers 1})]
      (tokenize-matches ctx))))


