(ns sanity-check
  (:require
   ;; [nextjournal.clerk :as clerk]
   ;; [nextjournal.clerk.viewer :as v]
   ;;[util.viewers :refer [wrap-seed]]
   [com.phronemophobic.llama :as llama]
   [com.phronemophobic.llama.util :as llutil]
   ;; com.phronemophobic.llama.raw-gguf
   [clojure.java.io :as io]
   [clojure.string :as str]))

(def llama7b-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin")
(def llama7b-gguf-path "models/llama-2-7b-chat.Q4_0.gguf")
(def gemma-2b-path "models/gemma-2b.gguf")
(def bge-large-path "models/bge-large-en-v1.5-q4_k_m.gguf")

;; (def llama7b-ggml (llama/create-context llama7b-path))
;; (def llama7b-ggml-embedding (llama/create-context llama7b-path
;;                                                   {:embedding true}))
;; (def llama7b-gguf (llama/create-context llama7b-gguf-path))
;; (def gemma (llama/create-context gemma-2b-path))
;; (def bge (llama/create-context bge-large-path {:embedding true}))

(def n-gpu-layers 1)
(def contexts
  (into
   {}
   (keep (fn [[k path opts]]
           (when (.exists (io/file path))
             [k (llama/create-context path opts)])))
   [[:llama7b-ggml llama7b-path {:n-gpu-layers n-gpu-layers}]
    [:llama7b-ggml-embedding llama7b-path {:embedding true :n-gpu-layers n-gpu-layers}]
    [:llama7b-gguf llama7b-gguf-path {:n-gpu-layers n-gpu-layers}]
    [:llama7b-gguf-embedding llama7b-gguf-path {:n-gpu-layers n-gpu-layers :embedding true}]
    [:gemma gemma-2b-path {:n-gpu-layers n-gpu-layers}]
    [:bge bge-large-path {:embedding true :n-gpu-layers n-gpu-layers}]]))


(def seed 4321)

(def embeddings
  (into {}
        (keep (fn [k]
               (when-let [ctx (get contexts k)]
                 (prn k)
                 (llama/llama-update ctx "banana" 0)
                 ;; use same normalization as llama.cpp embedding example.
                 [k (llutil/normalize-embedding (llama/get-embedding ctx))])))

        [:llama7b-ggml-embedding
         :llama7b-gguf-embedding
         :bge]))

(def responses
  (into {}
        (keep (fn [k]
                (when-let [ctx (get contexts k)]
                  (prn k)
                  (llama/set-rng-seed ctx seed)
                  [k (llama/generate-string
                      ctx
                      "Hello there"
                      {:seed seed
                       :samplef llama/sample-logits-greedy})])))

        [:llama7b-ggml
         :llama7b-gguf
         :gemma]))



(defn truncate [s n]
  (if (< (count s) n)
    s
    (subs s 0 n)))

;; TODO
;; automate comparison with llama.cpp/build/bin/embedding
(defn -main [& args]
  (doseq [[k response] responses]
    (println k)
    (println (truncate response 100)))
  (doseq [[k emb] embeddings]
    (println k)
    (println (truncate (pr-str (vec emb)) 100))))
