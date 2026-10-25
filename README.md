# awesomeRecs

This project explores the integration of the Reinforcement Learning into the recommender model training. 

- Architecture summary: 

    1. The baseline model uses gpt2 architecture and performs basic next-item prediction using item IDs as tokens, imitating natural language modeling approach.

    2. The reward model takes a candidate and previous history as input and produces the reward score based on the event (like/dislike/listen) and the percentage of track listen. 

    3. Reinforcement Learning model takes a history, a candidate, calculates the reward, and updates the policy weights.

- Dataset used: Yandex/Yambda-50M

- Number of users considered: 50.

- Vocabulary size: 1000 for the baseline gpt2 model and 10000 for the reward model. RL model uses the baseline vocabulary.

- Most important implementation files:

    1. baseline/data_preparation.ipynb: initial data preprocessing. 

    2. baseline/baseline_training.ipynb: a baseline gpt2 model implementation.

    3. reward/reward_model.ipynb: implementation of the reward model including multiple experiments with different versions.

    4. rl/rl_finetuning.ipynb: RL model implementation and evaluation. Experiments with different versions of the model.

- Artifact (data folder):

    1. bs_gpt2_rec: inital baseline model.

    2. bs_gpt2_rec_rl: a checkpoint for the first RL model.

    3. bs_gpt2_rec_rl_control, bs_gpt2_rec_rl_low_kl, bs_gpt2_rec_rl_low_kl_repeat_penalty

    4. artifacts.pkl: variables loaded from the initial data preparation pipeline

    5. reward_examples.pkl: input & target examples for train/validation/test splits.

    6. reward_interactions.pkl: extended data for the reward model (listens events).

    7. reward_model1_best.pt: a reward model that performed best.

## Research Conclusions

!important: check out the notebooks for more statistics, dataframes & summaries, if needed

1. Как меняется поведение модели после RL-дообучения? Удаётся ли максимизировать заданное вознаграждение? Чем рекомендации дообученной модели отличаются от рекомендаций исходной? Как можно объяснить эти изменения?

- Interestingly, in my experiments neither of the RL model showed clear improvement comparing to the baseline. Mean predicted reward increased by about 0.002 (reward ranging between -1 and 1), recall&10 & ndcg&10 even became a little bit worse. Recommendations of the RL models VS baseline model were also similar. 

I had three different RL models: the control one with defaut configs (bs_gpt2_rec_rl_control), the second one with the decreased value of KL, so the model is not tied up to the baseline too much (bs_gpt2_rec_rl_low_kl), and the third one where I used both lower KL & a penalty for repetitions. Neither one displayed any significant differences in behaviour. I think that the reason actually rises from my reward model formulation. I had several versions of it as well: the basic idea was to combine listening percentage and like/dislike influence, and other modifications danced around the same notion with different handlings of the dislike events. I tried using different heads for (1) the reward regression value and (2) the success in classifying dislikes from non-dislikes (because due to the dramatic lacking of the dislike events in the data, the model struggled to learn from them and could have eploited the misclassification (though there were no clear collapses)). In my opinion, one of main reasons for failure was my reward objective not matching the data I had. The objective was not inherently wrong (roughly, a big percentage and a like boosts the reward, a big percentage and a dislike significantly decreases it, etc.) but it was based on assumptions and failed to capture complex dynamics.

Also, perhaps it would have worked better if I based the reward calculation on the known target quality measure (like Petrov et al. did in ([1](https://arxiv.org/pdf/2403.04875))), maybe in my case likes/dislikes would be more suitable for reinforcement learning with human feedback (if we are using them directly in the reward objective). It can become a direction of future experiments.

| Model | Mean predicted reward | Median predicted reward | Mean training popularity | Median training popularity | Fraction with zero training occurrences | Repeat rate | Unique recommended items | Catalog coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.5487 | 0.5845 | 95.1613 | 76.0 | 0.0 | 0.3091 | 821 | 0.821 |
| rl_control | 0.5505 | 0.5864 | 95.6791 | 77.0 | 0.0 | 0.3087 | 824 | 0.824 |
| rl_low_kl | 0.5505 | 0.5865 | 95.6869 | 77.0 | 0.0 | 0.3087 | 824 | 0.824 |
| rl_low_kl_repeat_penalty | 0.5502 | 0.5863 | 95.3277 | 77.0 | 0.0 | 0.3086 | 822 | 0.822 |

| Model | Recall@10 | NDCG@10 | Repeat rate | Catalog coverage | Unique recommended items | Examples |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.2760 | 0.1702 | 0.3091 | 0.821 | 821 | 6724 |
| rl_control | 0.2751 | 0.1685 | 0.3087 | 0.824 | 824 | 6724 |
| rl_low_kl | 0.2748 | 0.1684 | 0.3087 | 0.824 | 824 | 6724 |
| rl_low_kl_repeat_penalty | 0.2739 | 0.1682 | 0.3086 | 0.822 | 822 | 6724 |

Top-rewarded items among baseline recommendations:

| token | item_id | mean_predicted_reward | recommendation_count | training_interactions |
| --- | --- | --- | --- | --- |
| 664 | 619494 | 0.974389 | 1 | 39 |
| 230 | 4211127 | 0.924082 | 22 | 69 |
| 347 | 1958145 | 0.900601 | 11 | 57 |
| 184 | 4334417 | 0.898944 | 70 | 76 |
| 179 | 555291 | 0.896413 | 73 | 77 |
| 150 | 1073534 | 0.889654 | 72 | 81 |
| 982 | 5603252 | 0.888960 | 12 | 32 |
| 235 | 4017613 | 0.887828 | 53 | 68 |
| 657 | 9180107 | 0.885566 | 14 | 39 |
| 183 | 4380055 | 0.883287 | 69 | 76 |
| 296 | 4944893 | 0.876948 | 65 | 62 |
| 163 | 7397709 | 0.876507 | 72 | 79 |
| 530 | 4053928 | 0.875901 | 26 | 45 |
| 72 | 1280625 | 0.874367 | 28 | 106 |
| 206 | 1851113 | 0.870954 | 15 | 72 |

Top-rewarded items among RL recommendations:

| token | item_id | mean_predicted_reward | recommendation_count | training_interactions |
| --- | --- | --- | --- | --- |
| 664 | 619494 | 0.974389 | 1 | 39 |
| 230 | 4211127 | 0.924847 | 23 | 69 |
| 657 | 9180107 | 0.906498 | 12 | 39 |
| 347 | 1958145 | 0.900601 | 11 | 57 |
| 179 | 555291 | 0.896330 | 75 | 77 |
| 184 | 4334417 | 0.891006 | 78 | 76 |
| 150 | 1073534 | 0.889654 | 72 | 81 |
| 183 | 4380055 | 0.889320 | 65 | 76 |
| 982 | 5603252 | 0.889045 | 11 | 32 |
| 235 | 4017613 | 0.887683 | 52 | 68 |
| 72 | 1280625 | 0.877858 | 27 | 106 |
| 163 | 7397709 | 0.877165 | 77 | 79 |
| 296 | 4944893 | 0.876983 | 64 | 62 |
| 654 | 2059641 | 0.875486 | 35 | 40 |
| 530 | 4053928 | 0.874581 | 27 | 45 |

2. Сохраняется ли релевантность рекомендаций? Оцените её с помощью Recall@K, NDCG@K и/или MAP@K. Проверьте, не начинает ли модель рекомендовать всем пользователям одни и те же объекты.

- The estimates can be seen above. The relevance was similar for both baseline and RL models (RL ones even showed a small decline). The repeat rate also did not change significantly and remained close to 31%, even for the model which explictly penalised repetitions.

- The coverage diagnostics did not show items variability collapse.

| Model | Recall@10 | NDCG@10 | Repeat rate | Catalog coverage | Unique recommended items | Examples |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.2760 | 0.1702 | 0.3091 | 0.821 | 821 | 6724 |
| rl_control | 0.2751 | 0.1685 | 0.3087 | 0.824 | 824 | 6724 |
| rl_low_kl | 0.2748 | 0.1684 | 0.3087 | 0.824 | 824 | 6724 |
| rl_low_kl_repeat_penalty | 0.2739 | 0.1682 | 0.3086 | 0.822 | 822 | 6724 |

3. Можно ли приблизиться к долгосрочным целям? Подумайте, как сформулировать вознаграждение, которое поощряло бы возвращение пользователей в сервис и продолжение взаимодействий. Какие сигналы в данных для этого подходят? Что можно проверить в офлайн-эксперименте, а для чего потребуется онлайн-проверка?

- I proposed the following rule for the reward calculation:

    1. User liked the track that he listened to for p%: 0.5 + 0.5p

    2. User disliked the track that he listened to for p%: -0.5 - 0.5p

    3. User listened to a track for p% without like/dislike: p

It was intended that dislikes would negatively influence the reward for the item, and for longer listening time the negative influence would be more significant (assuming the user will be more dissapointed if he/she wasted his/her time and still didn't like the song so much that even decided to dislike it. damn). So far, my hypothesis was not supported by the experiment results. Previous user activity, individual sessions durations, replaying previously recommended tracks, etc. can be utilised for offline experiments. On the other hand, hypothetically, for generating longer sequences it can be convenient to get continuosly updated human feedback, which will indeed be useful for long-term goals implementations. Interactive recommendation-feedback sessions can be created with analysing online queueing & smart shuffles activity. 

4. Что происходит при генерации нескольких шагов вперёд? Например, если последовательно сгенерировать пять объектов, каждый раз добавляя очередной объект в контекст. Такая последовательность может всё сильнее отличаться от реальной истории пользователя: сгенерированная рекомендация не означает, что пользователь действительно с ней взаимодействовал. Как это влияет на последующие рекомендации и оценку вознаграждения?

I did not experiment with the multi-step generation, but I suppose that this could accumulate errors in reward predictions and lead to unrealistic or repetitive sequences. An offline experiment could compare generated sequences with real user histories in terms of diversity/repetition rate/reward/etc.

5. Каков компромисс между релевантностью и вознаграждением? Можно ли увеличить вознаграждение без заметного снижения точности рекомендаций? Как на этот компромисс влияют функция вознаграждения и настройки RL-дообучения?

In my experiments, relevance and reward values did not provide enough information to conclude about their correlation. When I tried to let the model increase the reward by lowering KL, neither reward scores nor relevance metrics showed clear changes.

## Limitations

- The sample size was considerably small in order to make it managable under time & computing resources contraints (50 users and 1000 words vocabulary)

- The dataset contained several interesting events and fields that could have been utilized in the reward objective formulation (like is_organic field or unlike/undislike events)

- I did not experiment with multi-step sequences

- I intended to use dislikes information in my reward function but was not able to mitigate the imbalance of dislike events and non-dislike events

- There were no statistical tests performed

## Future directions

- Try to capture more complex dynamics in the reward model objective by firstly testing different conventional methods, such as pairwise preference, and then trying to formulate more general rule that can account for the lists of generated items (like in ([1](https://arxiv.org/pdf/2403.04875))).

- Try out different RL optimisations.

- Add statistical analysis.

- Try to increase the sample size & revise data preprocessing.
