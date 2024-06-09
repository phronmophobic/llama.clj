(ns com.phronemophobic.llama.impl.model
  (:refer-clojure :exclude [eval]))

(defprotocol ILLama
  (create-context [this model-path opts])
  (set-log-callback [this cb])
  (chat-apply-template [this template messages opts]))

(defprotocol ILLamaContext
  (token-eos [ctx])
  (token-bos [ctx])
  (token-is-eog [ctx token])
  (tokenize [ctx s add-bos?])
  (get-embedding [ctx])
  (get-logits [ctx])
  (sample-mirostat-v2 [ctx candidates-buf* mu* tau eta])
  (decode-token-to-str [ctx] [ctx opts])
  (decode-token-to-char [ctx] [ctx opts])
  (metadata [ctx])
  (model-description [ctx])
  (model-size [ctx])
  (model-n-params [ctx])
  (set-rng-seed [ctx seed])
  (n-ctx [ctx])
  (n-vocab [ctx])
  (eval
    [ctx s]
    [ctx s n-past]
    [ctx s n-past num-threads]))
