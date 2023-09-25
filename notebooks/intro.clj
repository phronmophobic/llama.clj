^{:nextjournal.clerk/visibility {:code :hide :result :hide}
  :nextjournal.clerk/toc true}
(ns intro
  (:require [nextjournal.clerk :as clerk]
            [nextjournal.clerk.viewer :as v]
            [instaparse.core :as insta]
            [util.viewers :refer [wrap-seed]]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [clojure.java.io :as io]
            [clojure.string :as str]))

^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(do
  (def llama7b-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin")
  (def llama7b-uncensored-path "models/llama2_7b_chat_uncensored.ggmlv3.q4_0.bin")
  (def llama-context (llama/create-context llama7b-path {:n-gpu-layers 1}))
  (def llama-uncensored-context (llama/create-context llama7b-uncensored-path {:n-gpu-layers 1}))
  (def seed 4321))

{:nextjournal.clerk/visibility {:code :show :result :show}}

;; # Intro to Running LLMs Locally

;; This guide covers the what, how, and why of running LLMs locally using 
;; [llama.clj](https://github.com/phronmophobic/llama.clj), a clojure wrapper for the [llama.cpp](https://github.com/ggerganov/llama.cpp) library.

;; Large language models (LLMs) are tools that are quickly growing in popularity. Typically, they are used via an API or service. However, many models are available to download and run locally even with modest hardware.

;; ## The One Basic Operation

;; From the perspective of using an LLM, there's really only one basic operation:

^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(def one-basic-description
  "Given a sequence of tokens, calculate the probability that a token will come next in the sequence. This probability is calculated for _all_ possible tokens.")
^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/md
 one-basic-description)

;; That's basically it. All other usage derives from this one basic operation.

;; ## Recreating the Chat Interface

;; If you've interacted with an LLM, it's probably while using one of the various chat interfaces. Before exploring other usages of local LLMs, we'll first explain how a chat interface can be implemented.

;; ### Tokens

;; Keen readers may have already noticed that chat interfaces work with text, but LLMs work with **tokens**. Choosing how to bridge the gap between text and tokens is an interesting topic for creating LLMs, but it's not important for understanding how to run LLMs locally. All we need to know is that text can be tokenized into tokens and vice versa. 

;; Just to get a sense of the differences between tokens and text, let's look at how the llama2 7b chat model tokenizes text.

^{:nextjournal.clerk/visibility {:code :show :result :hide}}
(def sentence "The quick brown fox jumped over the lazy dog.")
(def tokens
  (llutil/tokenize llama-context sentence))

;; One thing to notice is that are fewer tokens than letters:
(count tokens)
(count sentence)

;; If we untokenize each token, we can see that tokens are often whole words, but not always.

(mapv #(raw/llama_token_to_str llama-context %)
      tokens)

;; Just to get a feel for a typical tokenizer, we'll look at some basic stats.
^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(def token->str
  (into (sorted-map)
        (comp (map
               (fn [token]
                 [token (raw/llama_token_to_str llama-context token)]))
              (take-while (fn [[token untoken]]
                            untoken)))
        (range 0 Integer/MAX_VALUE)))

;; Number of tokens:
^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(count token->str)

;; The longest token:
^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(apply
 max-key
 (fn [[token untoken]]
   (count untoken))
 token->str)

;; Token with the most spaces:
^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(apply
 max-key
 (fn [[token untoken]]
   (count (filter #{\space} untoken )))
 token->str)

;; One last caveat to watch out for when converting between tokens and text is that not every token produces a valid utf-8 string. It may require multiple tokens before a valid utf-8 string is available. 

(def smiley-tokens (llutil/tokenize llama-context "😊"))

(def smiley-untokens (mapv #(raw/llama_token_to_str llama-context %)
                           smiley-tokens))

;; Fortunately, llama.clj has a utility for untokenizing that will take care of the issue:
(llutil/untokenize llama-context smiley-tokens)


;; ### Prediction

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/md
 (str "> " one-basic-description))

;; Returning to the one basic operation, we now know how to translate between text and tokens. Let's now turn to how prediction works.

;; While our description of the one basic operation says that LLMs calculates probabilities, that's not completely accurate. Instead,
;; LLMs calculate [logits](https://en.wikipedia.org/wiki/Logit) which are slightly different.
;; Even though logits aren't actually probabilities, we can mostly ignore the details except
;; to say that larger logits indicate higher probability and smaller logits indicate lower probability.

^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(defonce previous* (atom nil))
^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(defn get-logits [ctx s]
  (raw/llama_set_rng_seed ctx 1234)
  (cond

    (string? s)
    (llama/llama-update ctx s 0)

    (vector? s)
    (let [prev @previous*]
      (if (and
           (vector? prev)
           (= prev (butlast s)))
        (llama/llama-update ctx (last s))
        (do
          (llama/llama-update ctx (llama/bos) 0)
          (run! #(llama/llama-update ctx %)
                s)))))
  (reset! previous* s)

  (into [] (llama/get-logits ctx)))

;; Let's take a look at the logits for the prompt "Clojure is a".

(def clojure-is-a-logits
  (get-logits llama-context "Clojure is a"))

^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(comment

  ^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
  (def plot-data
    {:data [{:x (range (count clojure-is-a-logits))
             :y clojure-is-a-logits
             :type "scatter"}]
     :layout {:title "Logits"}})

  ^{:nextjournal.clerk/visibility {:code :hide :result :show}}
  (clerk/plotly
   plot-data
   )

  (apply max clojure-is-a-logits)
  (apply min clojure-is-a-logits)


  (defn softmax
    [values]
    (let [exp-values (mapv #(Math/exp %) values)
          sum-exp-values (reduce + exp-values)]
      (mapv #(/ % sum-exp-values) exp-values)))

  (apply + (softmax clojure-is-a-logits)))


;; `clojure-is-a-logits` is an array of numbers. The number of logits is 32,000 which is the number of tokens our model can represent. Each index in the array is proportional to the probability that the corresponding token will come next according to our LLM. 

;; Given that higher numbers are more probable, let's see what the top 10 candidates are:

(def highest-probability-candidates
  (->> clojure-is-a-logits
       ;; keep track of index
       (map-indexed (fn [idx p]
                      [idx p]))
       ;; take the top 10
       (sort-by second >)
       (take 10)
       (map (fn [[idx _p]]
              (llutil/untokenize llama-context [idx])))))

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/table
 (clerk/use-headers
  (into [["Highest Probability Candidates"]]
        (for [s highest-probability-candidates]
          ["Clojure is a" s]))))

;; And for comparison, let's look at the 10 least probable candidates:

(def lowest-probability-candidates
  (->> clojure-is-a-logits
       ;; keep track of index]
       (map-indexed (fn [idx p]
                      [idx p]))
       ;; take the bottom 10
       (sort-by second)
       (take 10)
       (map (fn [[idx _p]]
              (llutil/untokenize llama-context [idx])))))

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/table
 (clerk/use-headers
  (into [["Lowest Probability Candidates"]]
        (for [s lowest-probability-candidates]
          ["Clojure is a" s]))))


;; As you can see, the model does a pretty good job of finding likely and unlikely continuations.

;; #### Full Response Generation

;; Generating probabilities for the very next token is interesting, but not very useful by itself. What we really want is a full response. The way we do that is by using the probabilities to pick the next token, then append that token to our initial prompt, then retrieve new logits from our model, then rinse and repeat.

;; One of the decisions that most LLM APIs hide is the method for choosing the next token. In principle, we can choose any token and keep going (just as we were able to choose the initial prompt). The name for choosing the next token using the logits provided by the LLM is called **sampling**.

;; Choosing a sampling method is an interesting topic unto itself, but for now, we'll go with the most obvious method. We'll choose the token with the highest likelihood given by the model. Sampling using the highest likelihood option is called **greedy sampling**. Conventionally, greedy sampling isn't the best sampling method, but it's easy to understand and works well enough.

;; Ok, so we now have a plan for generating a full response:
;; 1. Feed out initial prompt into our model
;; 2. Sample the next token using greedy sampling
;; 3. Return to step #1 with the sampled token appended to our previous prompt

;; But wait! How do we know when to stop? LLMs define a token that llama.cpp calls end of sentence or eos for short (end of stream would be a more appropriate name, but oh well). We can repeat steps #1-3 until the eos token is the most likely token.

;; Finally, one last note before we generate a response is that chat models typically have a prompt format. The prompt format is a bit arbitrary and different models will have different prompt formats. Since the prompt format is defined by the model, users of models should check the documentation for the model being used.

;; Since, we're using llama2's 7b chat model, the prompt format is as follows:

(defn llama2-prompt
  "Meant to work with llama-2-7b-chat.ggmlv3.q4_0.bin"
  [prompt]
  (str
   "[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

" prompt " [/INST]
"))

;; Let's see how llama2 describes Clojure.

(def response-tokens
  (loop [tokens (llutil/tokenize llama-context
                                 (llama2-prompt "Describe Clojure in one sentence."))]
    (let [logits (get-logits llama-context tokens)
          ;; greedy sampling
          tok (->> logits
                   (map-indexed (fn [idx p]
                                  [idx p]))
                   (apply max-key second)
                   first)]
      (if (= tok (llama/eos))
        tokens
        (recur (conj tokens tok))))))

(def response
  (llutil/untokenize llama-context response-tokens))

;; See llama2's response below. Note that the response includes the initial prompt since the way we generate responses simply appends new tokens to the initial prompt.
;; However, most utilities in llama.clj strip the initial prompt since we're usually only interested in the answer generated by the LLM.

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/md
 (->> (str/split-lines response)
      (map #(str "> " %))
      (str/join "\n")))

;; Let's ask a follow up question. All we need to do is keep appending prompts and continue generating more tokens.

(def response-tokens2
  (loop [tokens
         (into response-tokens
               (llutil/tokenize llama-context
                                (str
                                 "[INST]"
                                 "Can I use it to write a web app?"
                                 "[/INST]"
                                 )))]
    (let [logits (get-logits llama-context tokens)
          ;; greedy sampling
          tok (->> logits
                   (map-indexed (fn [idx p]
                                  [idx p]))
                   (apply max-key second)
                   first)]
      (if (= tok (llama/eos))
        tokens
        (recur (conj tokens tok))))))

(def response2
  (llutil/untokenize llama-context response-tokens2))

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/md
 (->> (str/split-lines response2)
      (map #(str "> " %))
      (str/join "\n")))

;; We've now implemented a simple chat interface using the one basic operation that LLMs offer! To recap, LLMs work by calculating the likelihood of all tokens given a prompt. Our basic process for implementing the chat interface was:
;; 1. Feed our prompt into the LLM using the prompt structure specified by our chosen LLM.
;; 2. Sample the next token greedily and feed it back into the LLM.
;; 3. Repeated the process until we reached the end of sentence (eos) token.



;; ## Reasons for Running LLMs Locally

;; Now that we have a general sense of how LLMs work, we'll explore other ways to use LLMs and reasons for running LLMs locally rather than using LLMs through an API.

;; ### Privacy

;; One reason to run LLMs locally rather than via an API is making sure that sensitive or personal data isn't bouncing around the internet unnecessarily. Data privacy is important for both individual use as well as protecting data on behalf of users and customers.

;; ### Alternative Sampling Methods

;; Sampling is the method used for choosing the next token given the logits returned from an LLM. Our chat interface example used greedy sampling, but choosing the next token by always selecting the highest likelihood token often does not lead to the best results. The intuition for greedy sampling's poor performance is that always picking the highest probability tokens often leads to boring, uninteresting, and repetitive results.

;; Let's compare greedy sampling vs [mirostatv2](https://arxiv.org/abs/2007.14966), llama.clj's default sampling method:

(def prompt
  (llama2-prompt "What is the best ice cream flavor?"))

(def mirostat-response
  (llama/generate-string llama-context
                         prompt
                         {:seed 1234}))

;; **mirostatv2** response:
^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/md
 (->> (str/split-lines mirostat-response)
      (map #(str "> " %))
      (str/join "\n")))



(def greedy-response
  (llama/generate-string llama-context
                         prompt
                         {:samplef llama/sample-logits-greedy}))
;; **greedy** response:
^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/md
 (->> (str/split-lines greedy-response)
      (map #(str "> " %))
      (str/join "\n")))


;; Evaluating the outputs of LLMs is a bit of a dark art
;; which makes picking a sampling method difficult. Regardless, choosing or implementing the right sampling method
;; can make a big difference in the quality of the result.

;; To get a feel for how different sampling methods might impact results, check out the visualization tool at [https://perplexity.vercel.app/](https://perplexity.vercel.app/).

;; #### Constrained Sampling Methods

;; In addition to choosing sampling methods that improve responses, it's also possible to implement sampling methods that constrain the responses in interesting ways. Remember, it's completely up to the implementation as to determine which token gets fed back into the model.

;; ##### Run On Sentences

;; It's possible to arbitrarily select tokens. As an example, let's pretend we want our LLM to generate run-on sentences. We can artificially choose "and" tokens more often.

(def run-on-response
  (let [and-token (first (llutil/tokenize llama-context " and"))
        prev-tokens (volatile! [])]
    (llama/generate-string
     llama-context
     prompt
     {:samplef
      (fn [logits]
        (let [greedy-token (llama/sample-logits-greedy logits)
              ;; sort the logits in descending order with indexes
              top (->> logits
                       (map-indexed vector)
                       (sort-by second >))
              ;; find the index of the and token
              idx (->> top
                       (map first)
                       (map-indexed vector)
                       (some (fn [[i tok]]
                               (when (= tok and-token)
                                 i))))
              next-token
              ;; pick the and token if we haven't used it in the last
              ;; 5 tokens and if it's in the top 30 results
              (if (and (not (some #{and-token} (take-last 5 @prev-tokens)))
                       (< idx 30)
                       (not= (llama/eos) greedy-token))
                and-token
                greedy-token)]
          (vswap! prev-tokens conj next-token)
          next-token))})))

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/md
 (->> (str/split-lines run-on-response)
      (map #(str "> " %))
      (str/join "\n")))

;; By artificially boosting the chances of selecting "and", we were able to generate a rambling response. It's also possible to get rambling responses by changing the prompt to ask for a rambling response. In some cases, it's more effective to artificially augment the probabilities offered by the LLM.

;; ##### JSON Output

;; We can also use more complicated methods to constrain outputs. For example, we can force our
;; response to only choose tokens that satisfy a particular grammar.

;; In this example, we'll only choose tokens that produce valid JSON.

;; _Note: This example uses a subset of JSON that avoids sequences
;; that would require lookback to validate. Implementing lookback
;; to support arbitrary JSON output is left as an exercise for the reader._

(def json-parser
  (insta/parser (slurp
                 (io/resource "resources/json.peg"))))

(def json-response
  (let [prev-tokens (volatile! [])]
    (llama/generate-string
     llama-context
     (llama2-prompt "Describe some pizza toppings using JSON.")
     {:samplef
      (fn [logits]
        (let [sorted-logits (->> logits
                                 (map-indexed vector)
                                 (sort-by second >))
              first-jsonable
              (->> sorted-logits
                   (map first)
                   (some (fn [tok]
                           (when-let [s (try
                                          (llutil/untokenize llama-context (conj @prev-tokens tok))
                                          (catch Exception e))]
                             (let [parse (insta/parse json-parser s)
                                   toks (raw/llama_token_to_str llama-context tok)]
                               (cond
                                 ;; ignore whitespace
                                 (re-matches #"\s+" toks) false

                                 (insta/failure? parse)
                                 (let [{:keys [index]} parse]
                                   (if (= index (count s))
                                     ;; potentially parseable
                                     tok
                                     ;; return false to keep searching
                                     false))
                                 :else tok))))))]
          (vswap! prev-tokens conj first-jsonable)
          (if (Thread/interrupted)
            (llama/eos)
            first-jsonable)))})))

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/code
 json-response)




;; #### Classifiers


;; Another interesting use case for local LLMs is for quickly building simple classifiers. LLMs inherently keep statistics relating various concepts. For this example,
;; we'll create a simple sentiment classifier that describes a sentence as either "Happy" or "Sad". We'll also run our classifier against the llama2 uncensored model
;; to show how model choice impacts the results for certain tasks.

(defn llama2-uncensored-prompt
  "Meant to work with models/llama2_7b_chat_uncensored.ggmlv3.q4_0.bin"
  [prompt]
  (str "### HUMAN:
" prompt "
### RESPONSE:
"))

(defn softmax
  "Converts logits to probabilities. More optimal softmax implementations exist that avoid overflow."
  [values]
  (let [exp-values (mapv #(Math/exp %) values)
        sum-exp-values (reduce + exp-values)]
    (mapv #(/ % sum-exp-values) exp-values)))

;; Our implementation prompts the LLM to describe a sentence as either happy or sad using the following prompt:
;; ```clojure
;; (str "Give a one word answer of \"Happy\" or \"Sad\" for describing the following sentence: " sentence)
;; ```
;; We then compare the probability that the LLM predicts the response should be "Happy" vs
;; the probablility that the LLM predicts the response should be "Sad".

(defn happy-or-sad? [llama-context format-prompt sentence]
  (let [ ;; two tokens each
        [h1 h2] (llutil/tokenize llama-context "Happy")
        [s1 s2] (llutil/tokenize llama-context "Sad")

        prompt (format-prompt
                (str "Give a one word answer of \"Happy\" or \"Sad\" for describing the following sentence: " sentence))
        _ (llama/llama-update llama-context prompt 0)

        ;; check happy and sad probabilities for first tokens
        logits (llama/get-logits llama-context)
        probs (softmax logits)
        hp1 (nth probs h1)
        sp1 (nth probs s1)

        ;; check happy second token
        _ (llama/llama-update llama-context h1)
        logits (llama/get-logits llama-context)
        probs (softmax logits)
        hp2 (nth probs h2)

        ;; check sad second token
        _ (llama/llama-update llama-context s1
                              ;; ignore h1
                              (dec (raw/llama_get_kv_cache_token_count llama-context)))
        logits (llama/get-logits llama-context)
        probs (softmax logits)
        sp2 (nth probs s2)

        happy-prob (* hp1 hp2)
        sad-prob (* sp1 sp2)]
    {:emoji (if (> happy-prob sad-prob)
              "😊"
              "😢")
     ;; :response (llama/generate-string llama-context prompt {:samplef llama/sample-logits-greedy})
     :happy happy-prob
     :sad sad-prob
     :hps[hp1 hp2]
     :sps [sp1 sp2]}))

(def queries
  ["Programming with Clojure."
   "Programming with monads."
   "Crying in the rain."
   "Dancing in the rain."
   "Debugging a race condition."
   "Solving problems in a hammock."
   "Sitting in traffic."
   "Drinking poison."])

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/table
 (clerk/use-headers
  (into [["sentence" "llama2 sentiment" "llama2 uncensored sentiment"]]
        (for [sentence queries]
          (let [llama2-sentiment (happy-or-sad? llama-context llama2-prompt sentence)
                llama2-uncensored-sentiment (happy-or-sad? llama-uncensored-context llama2-uncensored-prompt sentence)]
            [sentence
             (:emoji llama2-sentiment)
             (:emoji llama2-uncensored-sentiment)])))))

;; In this example, the llama2 uncensored model vastly outperforms the llama2 model. It was very difficult to even find an example where llama2 would label a sentence as "Sad" due to its training. However, the llama2 uncensored model had no problem classifying sentences as happy or sad.

;; ### More Models Options

;; New models with different strengths, weaknesses, capabilities, and resource requirements are becoming available [regularly](https://huggingface.co/models?pipeline_tag=text-generation). 
;; As the classifier example showed, different models can perform drastically different depending on the task.

;; Just to give an idea, here's a short list of other models to try:

;; - [metharme-7b](https://huggingface.co/PygmalionAI/metharme-7b): This is an experiment to try and get a model that is usable for conversation, roleplaying and storywriting, but which can be guided using natural language like other instruct models.
;; - [GPT4All](https://github.com/nomic-ai/gpt4all): GPT4All is an ecosystem to train and deploy powerful and customized large language models that run locally on consumer grade CPUs.
;; - [OpenLLamMa](https://github.com/openlm-research/open_llama): we are releasing our public preview of OpenLLaMA, a permissively licensed open source reproduction of Meta AI’s LLaMA
;; - [ALMA](https://huggingface.co/haoranxu/ALMA-7B-Pretrain): ALMA (Advanced Language Model-based trAnslator) is an LLM-based translation model, which adopts a new translation model paradigm: it begins with fine-tuning on monolingual data and is further optimized using high-quality parallel data. This two-step fine-tuning process ensures strong translation performance.
;; - [LlaMa-2 Coder](https://huggingface.co/TheBloke/Llama-2-Coder-7B-GGUF): LlaMa-2 7b fine-tuned on the CodeAlpaca 20k instructions dataset by using the method QLoRA with PEFT library.

;; ## Conclusion

^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/md
 (str "> " one-basic-description))

;; LLMs really only have one basic operation which makes them easy to learn and easy to use.
;; Having direct access to LLMs provides flexibility in
;; cost, capability, and usage.

;; ## Next Steps

;; For more information on getting started, check out the [guide](https://phronmophobic.github.io/llama.clj/).


^{:nextjournal.clerk/visibility {:code :hide :result :show}}
(clerk/html
 [:div
  [:br]
  [:br]
  [:br]])

^{:nextjournal.clerk/visibility {:code :hide :result :hide}}
(comment
  (clerk/serve! {:watch-paths ["notebooks/intro.clj"]})
  
  (clerk/show! "notebooks/intro.clj")

  (clerk/build! {:paths ["notebooks/intro.clj"]
                 :out-path "docs/"})

  
  ,)
