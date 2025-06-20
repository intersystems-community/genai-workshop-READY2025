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


## Section 2 presentation

In Sections 3 and 4 we covered adding history and guardrails to our LLM application. 

How did we add these these two elements? (**I'm looking for an sanswer like we added them to the prompt**)

Taking a step back from this application, there is often a critical decision that has to be made when using LLMs, fortunately that is for most use cases quite easy to make. In our sample application, we are modifying the the prompt, the context, in order to refine the behavior of the LLM. 

What other techniques can we use to refine and optimize our LLM applications? (**Fine tuning and Agents(or tooling and mutliagents)**)

Fine tuning is a tricky business and bnot in scope for this workshop, but involves modfiying the parameters, the weights, biases and embedding vectors. In our last section, Sergei will introduce Agents. But for now, and for almost all of the starting work you do with an LLM, the focus will be primarily on modifying the prompt and context to get better results. Both of which are strings, the prompt a templated string, and the context the interpolated string.

Well, that makes things simple, right? Send in a string, get a string back. So even if I have a computer melded to my brain and communicating with a quantum computer to create a perfect representation of my memory, it still just has to output a string! Ok, I know multi-modal LLMs make it possible to send in audio and images, but my quantum computer only has 4 qbits so were are going to just stick with strings!

The only problem is that with even just considering 10k words, a small fraction of the total vocabulary, even if we limit our strings to 5 words that means our total possible combinations is 10 Vigintillion(10^20) inputs with the same amount of possible responses. Maybe not so easy!

This is why having a test first mindset is so important. While we used DeepEval to generate some test casts, using our gold standards, in a real application you will likely want to engage more with SME's. While this type of testing can be difficult because it crosses the tech and business lines and requires an SME's time, it is also what gives you a great advantage because you know your data better than anyone and that is the most important thing to have to build successful LLM apps.

Our testing example was really just an example and surely has some pretty big flaws.

Did anyone notice any issues? One was mentioned in the Conclusion.

** Too small of a RAG dataset **
** No specifying how many documents to return - default is 4 but this is buried in the code for iris-langchain**
** No specifying a minimal similiarity score **

Our next 2 topics - Tuning Retrieval and Agents will take youy through 2 more advanced techniques for refining your LLM enabled application.
