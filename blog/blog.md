# NHL Players Ranked By Attractiveness

### Backstory
Last year I read [an article](https://dailyhive.com/vancouver/ai-canucks-nhl-most-attractive-team) stating:

> Artificial intelligence has determined that the Canucks have the NHL's most attractive team, with an "average attractiveness score" of 7.841 out of 10.

The article, however, had several SIGNIFICANT flaws to it: 
1. The Vancouver Canucks have never come first in anything, so I immediately knew something was wrong
2. The article points to [Tonybet](https://tonybet.com/en/) as the source, but the link goes only to a generic betting homepage, with no study details or scoring method. That lack of transparency really grinds my gears.
3. The article claims it analyzed 1,079 NHL players, but that number doesn’t line up with a typical season snapshot (32 teams × 23 active roster spots ≈ 736). So who are these extra 343 ghosts that we analyzed... Sure I can understand a few called up from the minors, but THAT many??

So I did what any reasonable person would do: 

I built my own study and documented the whole thing end-to-end.

These are the questions I set out to answer:
1. Who are the most and least attractive players in the NHL?
2. Which NHL team is the most attractive on average?
3. Does attractiveness correlate with winning (or losing)?
4. Are defensemen more attractive than forwards—and are goalies really the ugliest?
5. Are Canadian NHL players more attractive than American NHL players & how do they compare to the rest of the world?

Bonus: How do the notorious Paul Bisonette and Ryan Whitney stack up in the rankings??

## Building

Before we can rank NHL headshots, we need a model that can look at a face and spit out an “attractiveness score”. And before we can train a model… we need training data.

## Get Data

So naturally I went hunting for a dataset where faces are already scored for attractiveness. Somehow, this exists?: [SCUT-FBP5500](https://www.kaggle.com/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores).

SCUT-FBP5500 is a facial beauty dataset from South China University of Technology with **5,500** neutral, frontal face images.

The basic idea is simple: each face is rated by a group of human judges, and the **average** of those ratings becomes the image’s “beauty score”. (It’s basically a very organized, peer-reviewed version of “rate me 1–5”.)

## Build ML Model

Now that we have a dataset with “beauty scores”, the goal is to train a model that can take a brand-new face (not in the dataset) and predict an attractiveness score on the same **1–5** scale.

Way back in university I built a baby CNN that could guess what number was written in a little pixel grid. I figured this was the same problem, but on steroids...

So I optimistically dusted off my old assignment and applied it to facial analysis, boy was I wrong. The initial MSE (Mean Square Error) was around 2 (keep in mind the scale is from 1-5). So it was great at randomly guessing attractiveness.

At that point I had two options:

1. Keep brute-forcing a CNN from scratch and pretend I had infinite data and infinite patience
2. Do what modern computer vision actually does: **reuse a model that already understands faces**, then train a much smaller model on top

So that’s what I did.

I switched to a transfer-learning style approach and leaned on pre-trained face models to do the heavy lifting. Instead of asking a CNN to learn “what a face is” from only a few thousand images, I first convert each face into a compact numerical representation (an embedding) using **ArcFace**.

Think of it like this:
- A raw image is a giant mess of pixels
- An embedding is a clean 512-number “summary” of the important facial features

Once I had those embeddings, predicting attractiveness became a standard regression problem and this is where things actually started working.

Then I tested three different models:

1. A **Neural Network** (4 dense layers + dropout)
2. **Ridge Regression** (linear, L2 regularization)
3. **SVR (RBF kernel)** with **GridSearchCV** tuning

Here’s what I found on the held-out test set:
- **Neural Network:** Test MSE **0.1895**, Test MAE **0.3322**, typical error **±0.435** (ouch)
- **Ridge Regression:** Test MSE **0.1080**, Test MAE **0.2534**, typical error **±0.329**
- **SVR (best):** Test MSE **0.0958**, Test MAE **0.2392**, typical error **±0.309**

So the winner is **SVR**. It’s non-linear (unlike Ridge), it plays nicely with the weird geometry of embedding space, and it was the only model that consistently got me below my “please be under 0.1 MSE” goal.

Now that we have a working model, we can run it on NHL player headshots and get our beauty scores

#### Geting Data
Getting NHL headshots took me a long time to navigate the murky waters of the internet. Out of a miracle I found https://github.com/Zmalski/NHL-API-Reference which has a beautiful API for NHL data, including player headshots and stats for active players. Thank you Zmalski, you’re my hero.

## Results & Analysis
Now that we have our model and our data, we can run the predictions and analyze the results.

### Goalies vs Defensemen v Forwards

First question: are goalies actually the ugliest, or is that just something we tell ourselves because they’re weird?

Turns out… goalies win. By average attractiveness score:

1. **Goalies** — **3.1272** (72 players)
2. **Defense** — **3.1135** (243 players)
3. **Forwards** — **3.1091** (457 players)

So yes, I’m officially reporting that *goalies are the most attractive position group*. I don’t make the rules, I just run the model.

### Most Attractive Countries

Now for the passport check. Here are the top countries by average attractiveness (minimums are just “who exists in the NHL”, so don’t bully the small sample sizes):

1. **Slovakia (SVK)** — **3.2267** (8)
2. **Denmark (DNK)** — **3.2197** (4)
3. **Czechia (CZE)** — **3.2102** (19)
4. **Norway (NOR)** — **3.1434** (2)
5. **Canada (CAN)** — **3.1331** (314)
6. **Sweden (SWE)** — **3.0990** (76)
7. **USA (USA)** — **3.0985** (222)

Canada is right near the top *and* has the biggest sample size by far, which is basically the statistical version of “yeah ok that checks out”.

### Most Attractive Teams (Unweighted)

If you simply average every player on each roster equally, the top teams come out like this:

1. **NSH** — **3.2387** (Points %: 52.1%)
2. **WPG** — **3.1910** (45.8%)
3. **NYI** — **3.1878** (60.2%)
4. **DAL** — **3.1782** (64.3%)
5. **COL** — **3.1760** (80.9%)

And… the Canucks?

- **VAN** — **3.0584** (Points %: 37.8%) — ranked **27th**

So if the internet told you Vancouver was #1… I have some bad news.

### Most Attractive Teams (Weighted by Ice Time)

One fair criticism of the “simple average” is that the 13th forward who plays 6 minutes a night counts the same as a top-pair defenseman playing 25.

So I also calculated a **weighted average** where players who actually play a lot of hockey count more (weighted by ice time). The top teams by that weighted score:

1. **NSH** — **3.2738** (Points %: 52.1%)
2. **SEA** — **3.2117** (53.1%)
3. **DAL** — **3.1594** (64.3%)
4. **WPG** — **3.1521** (45.8%)
5. **PIT** — **3.1463** (59.4%)

Vancouver moves slightly, but not enough to save the headline:

- **VAN** — **3.0526** (Points %: 37.8%) — ranked **26th**

### Does Attractiveness Correlate With Performance?

This is where I expected at least *some* funny trend (like “handsome guys take fewer penalties” or “ugly guys score more out of spite”).

Nope.

With **769 players** who had stats available, I ran both Pearson (linear) and Spearman (rank) correlations between attractiveness and common performance metrics:

- Goals: Pearson **-0.0663** (p=0.066), Spearman **-0.0419** (p=0.246)
- Assists: Pearson **-0.0497** (p=0.169), Spearman **-0.0495** (p=0.170)
- Points: Pearson **-0.0599** (p=0.097), Spearman **-0.0495** (p=0.170)
- Penalty minutes: Pearson **-0.0247** (p=0.495), Spearman **-0.0370** (p=0.306)

Translation: **there’s no meaningful relationship**. Being attractive doesn’t make you score more, win more, or take fewer trips to the penalty box.

### Biz & Whit: The Important Scientific Questions

And finally, the real reason we’re all here.

**Biz (biz.jpg)**
- Score: **3.59 / 5.0**
- Rank: **#45 / 807** (Top **5.6%**)

**Whit (whit.jpg)**
- Score: **3.37 / 5.0**
- Rank: **#156 / 807** (Top **19.3%**)

For comparison, some of the top of the leaderboard:
1. Vince Dunn — **4.05**
2. Jason Dickinson — **3.91**
3. Kris Letang — **3.88**
4. Marc Gatcomb — **3.86**
5. Scott Wedgewood — **3.85**
6. Anthony Duclair — **3.80**
7. Seth Jones — **3.78**
8. Eetu Luostarinen — **3.77**
9. Ross Colton — **3.77**
10. Jeremy Swayman — **3.76**
11. Mackie Samoskevich — **3.73**
12. Brady Skjei — **3.73**
13. Dmitry Kulikov — **3.73**
14. Vitek Vanecek — **3.73**
15. Luke Evangelista — **3.72**
16. William Karlsson — **3.71**
17. Denver Barkey — **3.71**
18. Spencer Stastney — **3.70**
19. Tyler Seguin — **3.70**
20. John Gibson — **3.68**

Biz is in elite company. Whit is still comfortably above average. The model has spoken.

