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

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean predicted reward</th>
      <th>Median predicted reward</th>
      <th>Mean training popularity</th>
      <th>Median training popularity</th>
      <th>Fraction with zero training occurrences</th>
      <th>Repeat rate</th>
      <th>Unique recommended items</th>
      <th>Catalog coverage</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Baseline</th>
      <td>0.5487</td>
      <td>0.5845</td>
      <td>95.1613</td>
      <td>76.0</td>
      <td>0.0</td>
      <td>0.3091</td>
      <td>821</td>
      <td>0.821</td>
    </tr>
    <tr>
      <th>rl_control</th>
      <td>0.5505</td>
      <td>0.5864</td>
      <td>95.6791</td>
      <td>77.0</td>
      <td>0.0</td>
      <td>0.3087</td>
      <td>824</td>
      <td>0.824</td>
    </tr>
    <tr>
      <th>rl_low_kl</th>
      <td>0.5505</td>
      <td>0.5865</td>
      <td>95.6869</td>
      <td>77.0</td>
      <td>0.0</td>
      <td>0.3087</td>
      <td>824</td>
      <td>0.824</td>
    </tr>
    <tr>
      <th>rl_low_kl_repeat_penalty</th>
      <td>0.5502</td>
      <td>0.5863</td>
      <td>95.3277</td>
      <td>77.0</td>
      <td>0.0</td>
      <td>0.3086</td>
      <td>822</td>
      <td>0.822</td>
    </tr>
  </tbody>
</table>
</div>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Recall@10</th>
      <th>NDCG@10</th>
      <th>Repeat rate</th>
      <th>Catalog coverage</th>
      <th>Unique recommended items</th>
      <th>Examples</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Baseline</th>
      <td>0.2760</td>
      <td>0.1702</td>
      <td>0.3091</td>
      <td>0.821</td>
      <td>821</td>
      <td>6724</td>
    </tr>
    <tr>
      <th>rl_control</th>
      <td>0.2751</td>
      <td>0.1685</td>
      <td>0.3087</td>
      <td>0.824</td>
      <td>824</td>
      <td>6724</td>
    </tr>
    <tr>
      <th>rl_low_kl</th>
      <td>0.2748</td>
      <td>0.1684</td>
      <td>0.3087</td>
      <td>0.824</td>
      <td>824</td>
      <td>6724</td>
    </tr>
    <tr>
      <th>rl_low_kl_repeat_penalty</th>
      <td>0.2739</td>
      <td>0.1682</td>
      <td>0.3086</td>
      <td>0.822</td>
      <td>822</td>
      <td>6724</td>
    </tr>
  </tbody>
</table>
</div>

Top-rewarded items among baseline recommendations:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>token</th>
      <th>item_id</th>
      <th>mean_predicted_reward</th>
      <th>recommendation_count</th>
      <th>training_interactions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>664</td>
      <td>619494</td>
      <td>0.974389</td>
      <td>1</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>230</td>
      <td>4211127</td>
      <td>0.924082</td>
      <td>22</td>
      <td>69</td>
    </tr>
    <tr>
      <th>2</th>
      <td>347</td>
      <td>1958145</td>
      <td>0.900601</td>
      <td>11</td>
      <td>57</td>
    </tr>
    <tr>
      <th>3</th>
      <td>184</td>
      <td>4334417</td>
      <td>0.898944</td>
      <td>70</td>
      <td>76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>179</td>
      <td>555291</td>
      <td>0.896413</td>
      <td>73</td>
      <td>77</td>
    </tr>
    <tr>
      <th>5</th>
      <td>150</td>
      <td>1073534</td>
      <td>0.889654</td>
      <td>72</td>
      <td>81</td>
    </tr>
    <tr>
      <th>6</th>
      <td>982</td>
      <td>5603252</td>
      <td>0.888960</td>
      <td>12</td>
      <td>32</td>
    </tr>
    <tr>
      <th>7</th>
      <td>235</td>
      <td>4017613</td>
      <td>0.887828</td>
      <td>53</td>
      <td>68</td>
    </tr>
    <tr>
      <th>8</th>
      <td>657</td>
      <td>9180107</td>
      <td>0.885566</td>
      <td>14</td>
      <td>39</td>
    </tr>
    <tr>
      <th>9</th>
      <td>183</td>
      <td>4380055</td>
      <td>0.883287</td>
      <td>69</td>
      <td>76</td>
    </tr>
    <tr>
      <th>10</th>
      <td>296</td>
      <td>4944893</td>
      <td>0.876948</td>
      <td>65</td>
      <td>62</td>
    </tr>
    <tr>
      <th>11</th>
      <td>163</td>
      <td>7397709</td>
      <td>0.876507</td>
      <td>72</td>
      <td>79</td>
    </tr>
    <tr>
      <th>12</th>
      <td>530</td>
      <td>4053928</td>
      <td>0.875901</td>
      <td>26</td>
      <td>45</td>
    </tr>
    <tr>
      <th>13</th>
      <td>72</td>
      <td>1280625</td>
      <td>0.874367</td>
      <td>28</td>
      <td>106</td>
    </tr>
    <tr>
      <th>14</th>
      <td>206</td>
      <td>1851113</td>
      <td>0.870954</td>
      <td>15</td>
      <td>72</td>
    </tr>
  </tbody>
</table>
</div>

Top-rewarded items among RL recommendations:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>token</th>
      <th>item_id</th>
      <th>mean_predicted_reward</th>
      <th>recommendation_count</th>
      <th>training_interactions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>664</td>
      <td>619494</td>
      <td>0.974389</td>
      <td>1</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>230</td>
      <td>4211127</td>
      <td>0.924847</td>
      <td>23</td>
      <td>69</td>
    </tr>
    <tr>
      <th>2</th>
      <td>657</td>
      <td>9180107</td>
      <td>0.906498</td>
      <td>12</td>
      <td>39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>347</td>
      <td>1958145</td>
      <td>0.900601</td>
      <td>11</td>
      <td>57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>179</td>
      <td>555291</td>
      <td>0.896330</td>
      <td>75</td>
      <td>77</td>
    </tr>
    <tr>
      <th>5</th>
      <td>184</td>
      <td>4334417</td>
      <td>0.891006</td>
      <td>78</td>
      <td>76</td>
    </tr>
    <tr>
      <th>6</th>
      <td>150</td>
      <td>1073534</td>
      <td>0.889654</td>
      <td>72</td>
      <td>81</td>
    </tr>
    <tr>
      <th>7</th>
      <td>183</td>
      <td>4380055</td>
      <td>0.889320</td>
      <td>65</td>
      <td>76</td>
    </tr>
    <tr>
      <th>8</th>
      <td>982</td>
      <td>5603252</td>
      <td>0.889045</td>
      <td>11</td>
      <td>32</td>
    </tr>
    <tr>
      <th>9</th>
      <td>235</td>
      <td>4017613</td>
      <td>0.887683</td>
      <td>52</td>
      <td>68</td>
    </tr>
    <tr>
      <th>10</th>
      <td>72</td>
      <td>1280625</td>
      <td>0.877858</td>
      <td>27</td>
      <td>106</td>
    </tr>
    <tr>
      <th>11</th>
      <td>163</td>
      <td>7397709</td>
      <td>0.877165</td>
      <td>77</td>
      <td>79</td>
    </tr>
    <tr>
      <th>12</th>
      <td>296</td>
      <td>4944893</td>
      <td>0.876983</td>
      <td>64</td>
      <td>62</td>
    </tr>
    <tr>
      <th>13</th>
      <td>654</td>
      <td>2059641</td>
      <td>0.875486</td>
      <td>35</td>
      <td>40</td>
    </tr>
    <tr>
      <th>14</th>
      <td>530</td>
      <td>4053928</td>
      <td>0.874581</td>
      <td>27</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>

2. Сохраняется ли релевантность рекомендаций? Оцените её с помощью Recall@K, NDCG@K и/или MAP@K. Проверьте, не начинает ли модель рекомендовать всем пользователям одни и те же объекты.

- The estimates can be seen above. The relevance was similar for both baseline and RL models (RL ones even showed a small decline). The repeat rate also did not change significantly and remained close to 31%, even for the model which explictly penalised repetitions.
- The coverage diagnostics did not show items variability collapse.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Recall@10</th>
      <th>NDCG@10</th>
      <th>Repeat rate</th>
      <th>Catalog coverage</th>
      <th>Unique recommended items</th>
      <th>Examples</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Baseline</th>
      <td>0.2760</td>
      <td>0.1702</td>
      <td>0.3091</td>
      <td>0.821</td>
      <td>821</td>
      <td>6724</td>
    </tr>
    <tr>
      <th>rl_control</th>
      <td>0.2751</td>
      <td>0.1685</td>
      <td>0.3087</td>
      <td>0.824</td>
      <td>824</td>
      <td>6724</td>
    </tr>
    <tr>
      <th>rl_low_kl</th>
      <td>0.2748</td>
      <td>0.1684</td>
      <td>0.3087</td>
      <td>0.824</td>
      <td>824</td>
      <td>6724</td>
    </tr>
    <tr>
      <th>rl_low_kl_repeat_penalty</th>
      <td>0.2739</td>
      <td>0.1682</td>
      <td>0.3086</td>
      <td>0.822</td>
      <td>822</td>
      <td>6724</td>
    </tr>
  </tbody>
</table>
</div>

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