(ns build
  (:require [clojure.tools.build.api :as b]
            [clojure.string :as str]))

(def lib 'com.phronemophobic/llama-cli)
(def version "0.1")
(def class-dir "target/classes")
(def basis* {:project "deps.edn"})
(def jar-file (format "target/%s-%s.jar" (name lib) version))
(def uber-file (format "target/%s-%s-standalone.jar" (name lib) version))

(defn clean [_]
  (b/delete {:path "target"}))


(defn compile [_]
  (b/compile-clj {:basis (b/create-basis basis*)
                  :ns-compile '[com.phronemophobic.llama-cli]
                  :class-dir class-dir
                  :jvm-opts ["-Dtech.v3.datatype.graal-native=true"
                             "-Dclojure.compiler.direct-linking=true"
                             "-Dclojure.spec.skip-macros=true"]})
  #_(b/javac {:src-dirs ["javasrc"]
              :class-dir class-dir
              :basis basis
                                        ;:javac-opts ["-source" "8" "-target" "8"]
              }))

(defn jar [opts]
  (compile opts)
  (b/write-pom {:class-dir class-dir
                :lib lib
                :version version
                :basis (b/create-basis basis*)
                :src-dirs ["src"]})
  (b/copy-dir {:src-dirs ["src" "resources"]
               :target-dir class-dir})
  (b/jar {:class-dir class-dir
          :jar-file jar-file}))

(defn uber [_]
  (clean nil)
  (b/copy-dir {:src-dirs ["src" "resources"]
               :target-dir class-dir})
  (compile nil)
  (b/uber {:class-dir class-dir
           :uber-file uber-file
           :basis (b/create-basis basis*)
           :main 'com.phronemophobic.llama-cli}))


