---
layout: post
title: Fine Tuning a Large Language Model for Socratic Interactions
---

[![Static Badge](https://img.shields.io/badge/Model%20-%20%40%20HuggingFace%20-blue?style=flat&logo=huggingface&logoSize=20px&color=blue&link=https%3A%2F%2Fhuggingface.co%2Feurecom-ds%2FPhi-3-mini-4k-socratic)](https://huggingface.co/eurecom-ds/Phi-3-mini-4k-socratic)
[![Static Badge](https://img.shields.io/badge/Model%20-%20%40%20Ollama%20-blue?style=flat&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAACgAAAA5CAYAAABEdGlTAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAACxMAAAsTAQCanBgAAAiASURBVGhD7ZhtaJ1nHcZPTtI2scli0tF0ypgyWFfndKUaV62CkymVyUTGnBsyvwymIpN9KEMpG4KK%2BkFlIMXVbvVDwW5MXeYKVYn7sGKb1s4NVjBa0yZrXprknDTN%2B5vX79%2FrefacNmle3E4%2B6AUn933%2FX6%2F79bnv5P6P%2F3m0tLSsa21trXfzbQMxie3myjAxMfHo1NTUKZVvqjw0Pj6%2B1aoVgxjEckxiP2rV8iDHH8zOzs5lMTMz0zcyMvJBmywb%2BBLD4QLkIJdNlgYF%2BoQcS9kZ6vWfbbZs4OswJSAXOW22OBToWfvOTU9Pt4%2BNjf3Mzeix2jttumTgk%2B0zMYntJh1%2F1qZXx9DQ0Hs1DQP2m9OaeRi5gh23iGAHwngZwMfudPo4MmJbxPIZIHcYZ5B3maK6uvqjFRUVjdTV46LWx4vUFeDXlED67V1dXXVuLgps8XEzjUVsclAnJ7mpZ3EFQTncIuOk%2FkZ9fX0X9cnJyVZ1dIp6Pp%2B%2FfsOGDTdSFyouXry4WaNxj%2FQP8dN6%2BoJG4%2F3W57DFhzoxiEWd2OSgTk5yU78qNPxPK0hA9b0W586dO3eN2v9GrkBzIvCASN0v2StqjyFLQF2jNCzdyxqle2T7lURPDGI5LPn2hkJQ%2FWmLF4bWyvO2nxsdHf2exQEl%2BatVEChmSS0Ek03XNDEuRbsEcljF2n7e4hRXTLF6kZ7u6v2AqwG1%2B11lmuuTpaDYkO9Vlel6Q%2FV%2BZAAb2caaBtkYIJsjmzvBFQQvwyUGhkZiyNUAJCQ7rCR3F4vFW1XfypeiUCjcKtl9SviyTVNcHkMoybEoFPQ5hhso2Q%2Bbm5srtYbe19HRwSfqpFVM1QBr0G4LQhviGyJ10W5M40liEVPqvM7D71vFGnzuktdVIKN9todEtwK2q2TBz1jMmhrSCH3SLotCu%2FzzijFld4jMEFOx%2F6FYnRYj32eXhaFR%2BY7t54UCM7Jft%2FmSoQ494RALgtw2nx8a7ttEII6SLNRLdW66oJJenjh79my1XZaMvr6%2BBmZDMQhGrHRGEpAbDnYJpAtUig%2BtXbv2kHbceyxiQZ%2BU0z6to1d6e3vPNzQ0NGpX9qpkxy4bw8PD1ytm44CwcePGpjVr1nyqqqrqocrKyi024TQ4p3w7a2pqXrMol9MObJDj6%2B5IQFOy%2B%2BjRo2ts8o5hcHCwToR%2BztJJABc42SR22o%2BtizWmKfi2VWWDZnB3lqQ4%2FSQU2u4M%2B6DlHAOLf27eISj3C6bBKPZcuHDhWnbtNy1DWJCQ82lVoEvGNo3iJFwYTXG7F9YtwU7QsKZXqtWClld66xa3X%2BVVfhiFSpR%2FCqsVQjEq%2BLm5Imjk%2Fugq3%2FDbKsRySlu9CoK6WWyvra0tuW3oK%2FBuFV%2FWb1w7%2B6DudmOhMDQtzTou7tbx8xGFieepOjqkeMcV%2Bvfr168%2FFoaGTpiadevW3asqZ%2BlvlC8urAm0xL4o2W%2B5ZChOJ71OMC32Jc%2FKzs7OKiX5A0rWhAg%2BZRWduRkCki9450KHDbZ244vyVOJCbHJYFdBd8U7thdBrDfbgMEwDJ%2B3oO20X6OnpaU6Mgep93d3d1bL7uOrnLeZY%2BpvW7091ED%2FMjzoyq%2FE7jw%2B%2BxLAY%2BRw5nC4gPg9ajf6fDOOrbjNC37VdQF%2BPrQRJoMQHNUU3SNZNW2WXfO5rb2%2B%2F4kBHhg4b23bjSwzagNjksEtA%2Bl9YzQgfRvCk24zEq%2Fp6lAy5kuzS71%2F6HVZvm%2BQUTwISyzd9xGuEGrnKi0Sjvg61FhOfx3qQxJcYxHLMXTYLaISvkW0HtkD2j3OCfyZZE5QK%2BC3bp9izZ0%2FcdLUmrpNNEVvVHwyloKA1%2Bp1QB0%2BrfFMxX7AqgK3jF4mBLImZhQg%2Fjh2Q7ZTibM2dOnWqUoH%2FYnmi%2BKx9SqCRuZ1O6FeQTXqp0E7Oq7f7HQLyJf%2FKwBYffIlhcQm0kT4nfbqexOkg8vyWLVtmFPxHkoWhtneVRjF9w2aR2AhM8YzrOR0lszoedinBiH4dx44de8KqgG3DOROjBMp5u3LHEwQbcXqSegh0IPJYp4pyQufaoWhcBl3HuP0WZdsgm7ssDtTV1U2rYIrP6OgoYYEtPvgSw%2BIScNUjN3W4iNMdodC1hoV5WsqAmKdn3XxQT5%2FBTiPVr6n8tMUB7chqyUous1pXrHFeeazvZyyeF%2BTGDsAJbqyPOxQghCqn9OXYZvt5oXPuBq2P2GmyH1XQX2r97FR5s0Q38aNu2V7ZTGCLD74OMy%2FIDQfs4QQ3dthjCICC%2FL2trW3RS6qOlFtkm33hUXALgQy%2FScsC2OJj9wVBbjjYjc32GAd1uvvU4%2F22XRT9%2Ff21mrJH5M%2B%2FPgY0JWOqj%2FOjjgwdNtjabVHAwXTo2H62d3rd0hTstt2yoKm4Tof0TXpMbeZHHZnVywIcTId%2FvbQw74fdntNR8YjtVg1wMJ05uGk352NrA72u3nqorBKyHOCW1%2Flz3u2c7mmbXV01ZDnAjU%2FUW%2B%2FPXG7bmTNnalwvO5w7PeaCmxbzx7Tj4kyg0MH6JevLDnKbClxm4ZY7cuQIl4W2kAqqn9CVacnHwtsFcpLbNODRBrdQ6qz6asIcqCcl%2F1ktB8jp9DGTcEIel4VCofA7yc9SBzL4gKtlQzYnXODkZlwo709G0OwfsKpsIGeWA5ysit3yUmgEzX1rU1PTf%2FW2XQk2bdpUQW7T4LP7Uii0OK%2FVt7MHIcy1Fr4WilUAuZNRhBPcGFquOHHVVjmmb2H6hi03yA0Hc5mBW15CPi2xWcR6UO%2BLPuqrAXLDwc3glhfLd4lwSFQWDxw4MBKNVQC54UAdTnDL6%2F7PWyJB9Y4dO0rexeWEc6dPhuCmO9f2ZGGqHNX2Tv9fXG6QGw7mwn1we15%2FTquRrDse4Kt2o3HuuKzAaXR09PR%2FABbe2vIkubrTAAAAAElFTkSuQmCC&logoSize=20px&color=blue&link=https%3A%2F%2Fhuggingface.co%2Feurecom-ds%2FPhi-3-mini-4k-socratic)
](https://ollama.com/eurecom-ds/phi-3-mini-4k-socratic)
[![Static Badge](https://img.shields.io/badge/Code%20-%20%40GitHub-blue?style=flat&logo=github)](https://github.com/GiovanniGatti/socratic-llm?tab=readme-ov-file#socratic-llm)

Large Language Models (LLMs) have shown outstanding ability in questioning-answering tasks. Yet, plain answers may not be the preferred outcome for education. Effective educators seek to encourage students to discover the answers to their questions by their means.
We inspire our work from the Socratic Method famously showcased in Plato's Republic. Here, Socrates questions Cephalus about the meaning of justice. When Cephalus defines justice as telling the truth and repaying debts, Socrates counters with a scenario that tests this idea, prompting Cephalus to review his definition.

We are fine-tuning LLMs to act more like Socratic tutors. Instead of just giving plain answers, our models ask probing questions to encourage students to think critically and deepen their understanding of the selected topic. For example, instead of providing the solution to $$x^2−4=0$$, the model prompts the student to factor the equation and find the values of  $$x$$ that solve it.

We use Direct Preference Optimization (DPO) to instruct LLMs to follow the Socratic Method. DPO is a widely used technique for aligning LLMs to human preferences. We generate diverse datasets and train the LLM to judge and rank answers based on their educational value. Our fine-tuned models perform much better than the originals, increasing their educational value.


## Evaluation metric
Defining the "socrativeness" of interactions is challenging. Therefore, we break the evaluation into four different aspects:

 - **questions**: a boolean score that should be *True* if the answer contains at least a question turned to the student, *False* otherwise.
 - **on topic**: a score on a scale from 1 (completely off-topic) to 5 (perfectly on-topic), which measures how much the teacher's answer is relevant to the ongoing conversation.
 - **helpful**: a score on a scale from 1 (completely unhelpful) to 5 (very helpful), which measures the usefulness of the response in providing guidance and support to the student.
 - **reveal answer**: a boolean score that should be *True* if the teacher's answer directly reveals the correct answer (which we want to penalize), *False* otherwise.

We ask a GPT-4o (i.e., LLM-as-a-judge) to evaluate interactions according to these four characteristics (check out the prompt [here](https://github.com/GiovanniGatti/socratic-llm/blob/main/templates/judge_llm.txt)). These four aspects are then uniformly weighed and normalized into a numerical *summary score*, ranging from zero to one.

To validate GPT-4o assessments, we compare them to those of human annotators for a set of 100 examples. We found a strong Pearson correlation ($$p=0.78$$) and aligned choices for the four components between GPT-4os and those of human annotators.

![_config.yml]({{ site.baseurl }}/images/human-vs-GPT-4o.svg){:height: 270px;vertical-align: middle;}
![_config.yml]({{ site.baseurl }}/images/humans-vs-GPT-4o-breakdown.svg){:width: 430px;vertical-align: middle;}

## Training Pipeline

There are many ways to steer LLMs to specific behaviors. Our choice of method is Direct Preference Optimization (DPO). DPO maximizes the likelihood of a user-defined set of "good" examples while minimizing the likelihood of "bad" examples. It has two main advantages: First, it has a low memory footprint, requiring just a reference model and the training model to execute; Secondly, it is a more stable algorithm and easier to tune than its counterparts.

As a choice of model, we fine-tune Phi-3-Mini-4k-Instruct, one of the small-size state-of-the-art LLMs (3.8 billion parameters). This model is already fine-tuned for the following instructions, thus simplifying the training procedure since we can request it to behave according to the Socratic method with prompt engineering. 

We follow the procedure illustrated below.

 - We generate with five candidate answers (Answer A to Answer E) for each input using this [prompt](https://github.com/GiovanniGatti/socratic-llm/blob/main/templates/inference.txt);
 - We use GPT-4o as a judge to assess the adherence of interactions to the Socratic method based on four criteria;
 - For each example, we extract a final summarized score ranging from zero to one, where one is the best outcome;
 - We select the best example (highest score) to be the accepted answer and the worst example (lowest score) as the rejected answer;
 - We perform training on the base model with DPO.

![_config.yml]({{ site.baseurl }}/images/training-pipeline.svg)

## Results & Analysis

We fine-tuned three models over three different datasets ([Debugging](https://arxiv.org/abs/2403.00199), [MathDial](https://arxiv.org/abs/2305.14536), and [TutorChat](https://arxiv.org/abs/2402.11111)). All these three datasets were designed to contain examples of high-quality Socratic interactions.

We observe that the model trained on TutorChat is the most
performing, yielding good performance on all three datasets. Notably, the TutorChat-trained model surpasses the models trained on MathDial and Debugging when evaluated on their respective test sets, albeit by a small margin. Such an effect is likely due to the preference dataset of TutorChat, which indicates a higher data diversity than the MathDial and Debugging datasets.

![_config.yml]({{ site.baseurl }}/images/table.png)

Below, we present the mean summary scores over the 100 samples for the TutorChat fine-tuned model and the base model using only prompt engineering. We add GPT-4o's performance with only prompt engineering to provide a reference of the best possible performance with prompt engineering-only strategies. The fine-tuned model improved significantly over the base model, reaching close performance to a much larger and more powerful GPT-4o in all datasets.

![_config.yml]({{ site.baseurl }}/images/perf-across-datasets.svg)

The TutorChat-trained model (our best model) showed significant gains in three key areas and now performs almost as well as GPT-4o. This also shows the model's strong generalization ability, as it was trained on TutorChat data but excelled on the different MathDial datasets.

![_config.yml]({{ site.baseurl }}/images/performance-breakdown.svg)

It also showed significant gains in three areas, nearing GPT-4o performance, and demonstrated strong generalization by excelling on a dataset different from its training data.

## Conclusions

Fine-tuning Large Language Models (LLMs) with Direct Preference Optimization enhances their performance in educational settings, especially when using the Socratic Method. These fine-tuned models better promote critical thinking by asking guiding questions instead of giving plain answers.

Future work will focus on using more powerful models and refining our metric system by adding more evaluation aspects. Larger models could improve the accuracy and effectiveness of Socratic dialogues, creating even more robust educational tools.

And do not forget to try out our model at [HuggingFace](https://huggingface.co/eurecom-ds/Phi-3-mini-4k-socratic) or [Ollama](https://ollama.com/eurecom-ds/phi-3-mini-4k-socratic)!