set -e
set -x


native-image --report-unsupported-elements-at-runtime \
             --features=clj_easy.graal_build_time.InitClojureClasses \
             --no-fallback \
             --trace-object-instantiation=org.objectweb.asm.Type \
             -H:ReflectionConfigurationFiles=config/reflect-config.json \
             -H:JNIConfigurationFiles=config/jni-config.json \
             -H:ResourceConfigurationFiles=config/resource-config.json \
             -jar ./target/*-standalone.jar \
             -H:Name=./target/llama-cli
