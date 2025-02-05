(ns com.phronemophobic.llama.raw-gguf-lib
  (:import com.sun.jna.Library
           com.sun.jna.Platform
           com.sun.jna.NativeLibrary))


(def ^:private RTLD_LOCAL 0)
(def ^:private RTLD_MEMBER 0x40000)
(def ^:private RTLD_LAZY 0x00001)

(def ^java.util.Map
  libllama-options
  (merge
   {com.sun.jna.Library/OPTION_STRING_ENCODING "UTF8"}
   (when (not (Platform/isWindows))
     {com.sun.jna.Library/OPTION_OPEN_FLAGS (bit-or
                                             RTLD_LOCAL
                                             RTLD_LAZY)})))
(def ^:no-doc libllama
  (delay
    (try
      ;; must be loaded in dependency order and before libllama.
      (doseq [libname ["ggml-base" "ggml-metal" "ggml-blas" "ggml-cpu" "ggml"]]
        (try
          (com.sun.jna.NativeLibrary/getInstance libname libllama-options)
          (catch UnsatisfiedLinkError e
            #_(println libname "not found."))))

      (com.sun.jna.NativeLibrary/getInstance "llama-gguf" libllama-options)
      (catch UnsatisfiedLinkError e
        ;; to support local builds
        (let [libllama (com.sun.jna.NativeLibrary/getInstance "llama" libllama-options)]
          ;; Make sure it's not an old version
          (try
            (.getFunction ^com.sun.jna.NativeLibrary libllama
                          "llama_token_to_piece")
            (catch UnsatisfiedLinkError _
              ;; throw original error
              (throw e)))
          libllama)))))
