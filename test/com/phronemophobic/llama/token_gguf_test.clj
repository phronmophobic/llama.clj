(ns com.phronemophobic.llama.token-gguf-test
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
            )
  (:import com.sun.jna.Pointer))


(defn token-roundtrip-alphanumeric [ctx]
  (prop/for-all [s gen/string-alphanumeric]
                (= s
                   (->> s
                        (llutil/tokenize ctx)
                        (llutil/untokenize ctx)))))

(defn token-roundtrip [ctx]
  (prop/for-all [s gen/string]
                (= s
                   (->> s
                        (llutil/tokenize ctx)
                        (llutil/untokenize ctx)))))


(comment

  (def model-path "models/mistral-7b-instruct-v0.1.Q4_0.gguf")
  (def model-path "models/llama-2-7b-chat.Q4_0.gguf")
  (def ctx (llama/create-context model-path))

  (tc/quick-check 10000 (token-roundtrip-alphanumeric ctx))
  (tc/quick-check 10000 (token-roundtrip ctx))

  ,)

(defspec token-llama2-roundtrip-alphanumeric-spec
  10000
  (let [model-path "models/llama-2-7b-chat.Q4_0.gguf"]
    (let [ctx (llama/create-context model-path {:n-gpu-layers 1})]
      (token-roundtrip-alphanumeric ctx))))

(defspec token-mistral-roundtrip-alphanumeric-spec
  10000
  (let [model-path "models/mistral-7b-instruct-v0.1.Q4_0.gguf"]
    (let [ctx (llama/create-context model-path {:n-gpu-layers 1})]
      (token-roundtrip-alphanumeric ctx))))

(defspec token-llama2-roundtrip-spec
  10000
  (let [model-path "models/llama-2-7b-chat.Q4_0.gguf"]
    (let [ctx (llama/create-context model-path {:n-gpu-layers 1})]
      (token-roundtrip ctx))))

(defspec token-mistral-roundtrip-spec
  10000
  (let [model-path "models/mistral-7b-instruct-v0.1.Q4_0.gguf"]
    (let [ctx (llama/create-context model-path {:n-gpu-layers 1})]
      (token-roundtrip ctx))))
