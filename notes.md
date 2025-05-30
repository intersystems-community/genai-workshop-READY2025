## Issues encountered with Environment
1. Codespaces seems very finicky and unpredictable in certain features
   1. Port forwarding 
      1. Sometimes works and sometimes not with same config
      2. With different config but same port forwarding will not work
      3. Even when port is forwarded sometimes service seems to not be available
   2. Sometime what works in VsCode as DevContainer does not work in Codespaces - which means you have to change and deploy to troubleshoot and that takes at least 5 minutes. Since there is really no docs on why this should happen this process is partially trial and error making it a very slow process.
2. DevContainers, Docker-compose and docker 
   1. All have implicit configuration dependencies that are very hard to know - lots of magic strings that need to work together across the config files. For example, the volume mount in docker compose must match the workspaceMount.target and usually workspaceFolder in .devcontainer.json
   2. Rebuilding with --no-cache with any change except for in docker-compose.yml is usually necessary
3. IRIS python
   1. Official db-api drivers might not work with many of the other community libs like iris-alchemy, iris-llama, iris-langchain
   2. Official driver throws ssl error
   3. Embedding.Config fails because it does not seem that sentence_transformers is installed.
      1. INSERT INTO %Embedding.Config (Name, Configuration, EmbeddingClass, Description)
         VALUES ('sentence-transformers-config',
                  '{"modelName":"sentence-transformers/all-MiniLM-L6-v2",
                     "hfCachePath":"/Users/InterSystems/VEC147/hfCache",
                     "maxTokens": 256,
                     "checkTokenCount": true}',
                  '%Embedding.SentenceTransformers',
                  '',
                  'a small SentenceTransformers embedding model')
      2. When running this on the IRIS host(not if running from another container), I get this error:
         1. Unrecognized model in sentence-transformers/all-MiniLM-L6-v2. Should have a `model_type` key in its config.json, or contain one of the following strings in its name: albert, align, altclip, aria, aria_text, audio-spectrogram-transformer, autoformer, aya_vision, bamba, bark, bart, beit, bert, bert-generation, big_bird, bigbird_pegasus, biogpt, bit, blenderbot, blenderbot-small, blip, >]

   4. Install order matters with iris python libs(and it should not)
       > [python 8/8] RUN python create_desc_vectors.py:
       Traceback (most recent call last):
         File "/home/python/work/create_desc_vectors.py", line 6, in <module>
           import iris
         File "/usr/local/lib/python3.13/site-packages/iris/__init__.py", line 10, in <module>
               raise Exception("""Cannot find InterSystems IRIS installation directory
           Please set IRISINSTALLDIR environment variable to the InterSystems IRIS installation directory""")
       Exception: Cannot find InterSystems IRIS installation directory
           Please set IRISINSTALLDIR environment variable to the InterSystems IRIS installation directory
4. Docker 
   1. Does not track all the files for IRIS, like class files, so often need to rerun full build using --no-cache, which is extra slow becauase sentence-transformers is not included in iris
   2. Docker stalling on builds after very short time and requires restart
      1. Turing off resource saver to try and fix. Trying to apply and restart even stalled


## Notes
1. Using the db-api driver from the community
   1. https://github.com/intersystems-community/intersystems-irispython/releases/download/3.9.2/intersystems_iris-3.9.2-py3-none-any.whl
2. Existing data load uses pandas, which creates the GenAI.encounters table from the CSV
   