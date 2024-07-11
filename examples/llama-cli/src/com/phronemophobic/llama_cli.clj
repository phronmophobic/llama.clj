(ns com.phronemophobic.llama-cli
  (:require [com.phronemophobic.llama :as llama]
            com.phronemophobic.llama.raw-gguf)
  (:gen-class))

(defn -main [model-path prompt]
  (llama/-main model-path prompt))
