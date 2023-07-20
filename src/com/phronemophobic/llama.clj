(ns com.phronemophobic.llama
  (:require [com.phronemophobic.llama.raw :as raw])
  (:import java.lang.ref.Cleaner
           com.sun.jna.Memory
           com.sun.jna.Pointer
           com.sun.jna.ptr.IntByReference
           com.sun.jna.ptr.FloatByReference
           com.sun.jna.Structure))

(raw/import-structs!)

(def token-data-size (.size (llama_token_data.)))

(defn ^:private generate-tokens [ctx ^Memory token-buf num-tokens]
  (let [max-context-size (raw/llama_n_ctx ctx)]
    (loop [num-tokens num-tokens
           candidates-buf nil]
      (let [token-count (raw/llama_get_kv_cache_token_count ctx)]
        (if (< token-count max-context-size)
          (do
            (raw/llama_eval ctx token-buf num-tokens token-count 1)
            (let [n-vocab (raw/llama_n_vocab ctx)
                  logits (-> ^FloatByReference (raw/llama_get_logits ctx)
                             .getPointer
                             (.getFloatArray 0 n-vocab))

                  buf-size (* token-data-size n-vocab)
                  candidates-buf (if (and candidates-buf
                                          (>= (.size ^Memory candidates-buf)
                                              buf-size))
                                   candidates-buf
                                   (Memory. buf-size))]
              (doseq [i (range n-vocab)]
                (let [base-addr (* i token-data-size)
                      id i
                      logit (aget logits id)
                      p 0]
                  (.setInt candidates-buf base-addr id)
                  (.setFloat candidates-buf (+ base-addr 4) logit)
                  (.setFloat candidates-buf (+ base-addr 8) 0)))
              (let [candidates-array-head (doto (Structure/newInstance llama_token_dataByReference
                                                                       candidates-buf)
                                            (.read))
                    candidates* (doto (llama_token_data_arrayByReference.)
                                  (.writeField "data" candidates-array-head)
                                  (.writeField "size" (long n-vocab))
                                  (.writeField "sorted" (byte 0)))

                    new-token-id (raw/llama_sample_token_greedy ctx candidates*)]
                ;; (prn new-token-id (llama_token_to_str ctx new-token-id))
                (print (raw/llama_token_to_str ctx new-token-id))
                (flush)
                (when (not= new-token-id
                            (raw/llama_token_eos))
                  (.setInt token-buf 0 new-token-id)
                  (recur 1
                         candidates-buf))))))))))



(defn ^:private llm-prompt [model-path prompt]
  (raw/llama_backend_init 0)
  (let [params (doto ^llama_context_params (raw/llama_context_default_params)
                 ;; (.writeField "n_gpu_layers" (int 1))
                 )
        model (raw/llama_load_model_from_file model-path params)
        _(assert model)
        context (raw/llama_new_context_with_model model params)
        ;; // Add a space in front of the first character to match OG llama tokenizer behavior
        prompt (str " " prompt)
        add-bos 1
        ;; tokens are ints
        num-tokens (* 4 (+ add-bos (count prompt)))
        token-buf (Memory. num-tokens)
        embd_inp (raw/llama_tokenize context prompt token-buf num-tokens add-bos)

        n-ctx (raw/llama_n_ctx context)


        _ (assert (< embd_inp (- n-ctx 4)) "prompt too long")

        ;; // do one empty run to warm up the model
        #_(let [tmp (IntByReference. (raw/llama_token_bos))]
            (raw/llama_eval context tmp 1 0 1))

        ;; (e arr (.toArray (llama_token_dataByReference.) 3) )


        max-context-size n-ctx]
    (generate-tokens context token-buf embd_inp)))


(defn -main [model-path prompt]
  ;; "/Users/adrian/workspace/llama.cpp/models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q4_0.bin"
  (llm-prompt model-path prompt))
