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


(defonce ^:private llm-init
  (delay
    (raw/llama_backend_init 0)))
(defn create-context [model-path]
  @llm-init
  (let [params (doto ^llama_context_params (raw/llama_context_default_params)
                 (.writeField "n_gpu_layers" (int 1))
                 )
        model (raw/llama_load_model_from_file model-path params)
        _(assert model)
        context (raw/llama_new_context_with_model model params)]
    context))


;; todo use a soft cache
(defonce ^:private
  token-bufs
  (atom {}))

(defn ^:private get-token-buf [ctx n]
  (get
   (swap! token-bufs
          (fn [m]
            (let [buf (get m ctx)]
              (if (and buf
                       (>= (.size ^Memory buf)
                           (* 4 n)))
                m
                (assoc m ctx (Memory. (* 4 n)))))))
   ctx))

(defn ^:private tokenize [ctx s add-bos?]
  (let [add-bos (if add-bos?
                  1
                  0)

        s (if add-bos?
            (str " " s)
            s)
        max-tokens (+ add-bos (count s))
        token-buf (get-token-buf ctx max-tokens)
        num-tokens (raw/llama_tokenize ctx s token-buf max-tokens add-bos)]
    [num-tokens token-buf]))

(defn llama-update
  ([ctx s]
   (llama-update ctx s (raw/llama_get_kv_cache_token_count ctx)))
  ([ctx s n-past]
   (let [[num-tokens token-buf]
         (cond
           (string? s)
           (tokenize ctx s (zero? n-past))

           (integer? s)
           (let [^Memory buf (get-token-buf ctx 1)]
             [1 (doto buf
                  (.setInt 0 s))]))]
     (raw/llama_eval ctx token-buf num-tokens n-past 1)
     ctx)))

(defn sample-logits-greedy [logits]
  (transduce (map-indexed vector)
             (completing
              (fn [[idx1 f1 :as r1] [idx2 f2 :as r2]]
                (if (> f1 f2)
                  r1
                  r2))
              first)
             [nil Float/MIN_VALUE]
             logits))

(defn get-logits [ctx]
  (let [n-vocab (raw/llama_n_vocab ctx)]
   (-> ^FloatByReference (raw/llama_get_logits ctx)
       .getPointer
       (.getFloatArray 0 n-vocab))))

(comment
  (def model-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin")

  (def ctx (create-context model-path))

  (def prompt "In the context of llms, what is a logit?")
  ;; updates context logits
  (llama-update ctx prompt)

  (def results
    (loop [results [prompt]]
      (let [next-token (sample-logits-greedy (get-logits ctx))
            next-str (raw/llama_token_to_str ctx next-token)]
        (print next-str)
        (flush)
        (if (not= next-token
                  (raw/llama_token_eos))
          (do
            (llama-update ctx next-token)
            (recur (conj results next-str)))
          results))))

  ,)

(defn -main [model-path prompt]
  ;; "/Users/adrian/workspace/llama.cpp/models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q4_0.bin"
  (llm-prompt model-path prompt))
