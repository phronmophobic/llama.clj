(ns build
  (:require [clojure.tools.build.api :as b]
            [clojure.string :as str]))

(def lib 'com.phronemophobic/llama-clj)
(def version "0.8.6-beta1")

(def class-dir "target/classes")
(def basis (b/create-basis {:project "deps.edn"}))
(def jar-file (format "target/%s-%s.jar" (name lib) version))
(def src-pom "./pom-template.xml")

(defn clean [_]
  (b/delete {:path "target"}))

(defn compile [_]
  #_(b/javac {:src-dirs ["src-java"]
            :class-dir class-dir
            :basis basis
            :javac-opts ["-source" "8" "-target" "8"]}))

(defn jar [opts]
  (compile opts)
  (b/write-pom {:class-dir class-dir
                :src-pom src-pom
                :lib lib
                :version version
                :basis basis
                :src-dirs ["src"]})
  (b/copy-dir {:src-dirs ["src" "resources"]
               :target-dir class-dir})
  (b/jar {:class-dir class-dir
          :jar-file jar-file}))

(defn deploy [opts]
  (jar opts)
  (try ((requiring-resolve 'deps-deploy.deps-deploy/deploy)
        (merge {:installer :remote
                :artifact jar-file
                :pom-file (b/pom-path {:lib lib :class-dir class-dir})}
               opts))
       (catch Exception e
         (if-not (str/includes? (ex-message e) "redeploying non-snapshots is not allowed")
           (throw e)
           (println "This release was already deployed."))))
  opts)

(defn deploy-combined [opts]
  (let [ggml-version "6e88a462d7d2d281e33f35c3c41df785ef633bc1"
        ggml-deps
        {'com.phronemophobic.cljonda/llama-cpp-darwin-aarch64 {:mvn/version ggml-version}
         'com.phronemophobic.cljonda/llama-cpp-darwin-x86-64 {:mvn/version ggml-version}
         'com.phronemophobic.cljonda/llama-cpp-linux-x86-64 {:mvn/version ggml-version}}

        gguf-version "b4634"
        gguf-deps
        {'com.phronemophobic.cljonda/llama-cpp-gguf-linux-x86-64 {:mvn/version gguf-version}
         'com.phronemophobic.cljonda/llama-cpp-gguf-darwin-aarch64 {:mvn/version gguf-version}
         'com.phronemophobic.cljonda/llama-cpp-gguf-darwin-x86-64 {:mvn/version gguf-version}}
        basis (b/create-basis {:project
                               {:deps
                                (merge
                                 {lib {:mvn/version version}}
                                 ggml-deps
                                 gguf-deps)}})
        combined-lib 'com.phronemophobic/llama-clj-combined]
    (clean opts)
    (b/write-pom {:class-dir class-dir
                  :src-pom src-pom
                  :lib combined-lib
                  :version version
                  :basis basis})
    (b/jar {:jar-file jar-file
            :class-dir class-dir})
    (try ((requiring-resolve 'deps-deploy.deps-deploy/deploy)
          (merge {:installer :remote
                  :artifact jar-file
                  :pom-file (b/pom-path {:lib combined-lib
                                         :class-dir class-dir})}
                 opts))
         (catch Exception e
           (if-not (str/includes? (ex-message e) "redeploying non-snapshots is not allowed")
             (throw e)
             (println "This release was already deployed."))))))

;; doesn't work
;; need to lazy load native lib
(defn uber [_]
  (let [basis (b/create-basis {:project "deps.edn"
                               :aliases [:mvn-llama
                                         :native-image]})]
    (clean nil)
    (b/copy-dir {:src-dirs ["src" "resources"]
                 :target-dir class-dir})
    (b/compile-clj {:basis basis
                    :src-dirs ["src"]
                    :compile-opts {:direct-linking true}
                    :ns-compile ['com.phronemophobic.llama.cli]
                    :class-dir class-dir})
    (b/uber {:class-dir class-dir
             :uber-file "target/uber.jar"
             :basis basis
             :main 'com.phronemophobic.llama.cli})))
