(ns util.viewers
  (:require [nextjournal.clerk :as clerk]))

(defmacro wrap-seed
  ([form]
   `(wrap-seed ~form nil))
  ([form visibility]
   `(do
      (raw/llama_set_rng_seed llama-context seed)
      (let [~'result ~form]
        {::llama-view
         {:form (quote ~form)
          :result ~'result
          :visibility ~visibility}}))))

(defmacro wrap-seed
  ([form]
   `(wrap-seed ~form nil))
  ([form visibility]
   `(do
      (raw/llama_set_rng_seed ~'llama-context ~'seed)
      (let [~'result ~form]
        {::llama-view
         {:form (quote ~form)
          :result ~'result
          :visibility ~visibility}}))))

(def llama-viewer
  {:pred ::llama-view
   :transform-fn
   (clerk/update-val
    (fn [{m ::llama-view}]

      (let [blocks []
            
            form (:form m)
            result (:result m)

            visibility (or (:visibility m)
                           {:code :show :result :show})
            blocks (if (= :show
                          (:code visibility))
                     (conj blocks
                           (clerk/md
                            (str "```clojure\n"

                                 (with-out-str
                                   (clojure.pprint/pprint form))
                                 "\n```")))
                     blocks)

            blocks (if (= :show
                          (:result visibility))
                     (conj blocks
                           (clerk/md
                            (str "```clojure\n"
                                 result
                                 "\n```")))
                     blocks)]
        
        (apply
         clerk/col
         {::clerk/width :wide}
         blocks))))})

