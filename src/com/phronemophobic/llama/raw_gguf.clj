(ns com.phronemophobic.llama.raw-gguf
  (:require [com.phronemophobic.llama.raw-gguf-lib :as libllama]))

(defn guess-model-api []
  (let [lib @libllama/libllama]
    (try
      (.getFunction ^com.sun.jna.NativeLibrary lib
                    "llama_set_rng_seed")
      @(requiring-resolve 'com.phronemophobic.llama.raw-gguf-b3040/llama-model)
      (catch UnsatisfiedLinkError e
        @(requiring-resolve 'com.phronemophobic.llama.raw-gguf-b4634/llama-model)))))



