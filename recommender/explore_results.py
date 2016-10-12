"""
Calculate the validation scores for all models
"""
import pandas as pd
from recommender.validation import get_validation_scores_per_model,\
                                   get_validation_score_combine_var,\
                                   get_validation_score_combine_simple
import recommender.io as io
import os

pd.options.display.width = 150
pd.options.mode.chained_assignment = None

# Paths of results
directory = r"data\predictions"
per_user_result_files = [os.path.join(directory, file) for file in os.listdir(directory)
                         if file.endswith(".csv") and file.startswith("per_user")]

per_movie_result_files = [os.path.join(directory, file) for file in os.listdir(directory)
                          if file.endswith(".csv") and file.startswith("per_movie")]

ratings = io.get_rating()
ratings_test = ratings.query("test == True")
ratings_train = ratings.query("test == False")

# per_movie models
per_movie_scores = []
for movie_path in per_movie_result_files:
    per_movie_result = pd.merge(ratings_test, pd.read_csv(movie_path))
    per_movie_result = per_movie_result[~per_movie_result.movie_est.isnull()]
    per_movie_result["movie_est"][per_movie_result.movie_est > 5] = 5
    per_movie_result["movie_est"][per_movie_result.movie_est < 1] = 1
    scores = get_validation_scores_per_model(per_movie_result["rating"],
                                             per_movie_result["movie_est"])
    scores["est_avg_var"] = per_movie_result.movie_est_var.mean()
    scores.name = "mov_" + os.path.basename(movie_path)[10:-9]
    per_movie_scores.append(scores)

per_movie_scores = pd.DataFrame(per_movie_scores)

# per_user models
per_user_scores = []
for user_path in per_user_result_files:
    per_user_result = pd.merge(ratings_test, pd.read_csv(user_path))
    per_user_result = per_user_result[~per_user_result.user_est.isnull()]
    per_user_result["user_est"][per_user_result.user_est > 5] = 5
    per_user_result["user_est"][per_user_result.user_est < 1] = 1
    scores = get_validation_scores_per_model(per_user_result["rating"],
                                             per_user_result["user_est"])
    scores.name = "usr_" + os.path.basename(user_path)[9:-9]
    per_user_scores.append(scores)

per_user_scores = pd.DataFrame(per_user_scores)

# combined model
per_combi_scores = []
for user_path in per_user_result_files:
    per_user_result = pd.merge(ratings_test, pd.read_csv(user_path))
    per_user_result["user_est"] = per_user_result["user_est"]
    per_user_result["user_est"][per_user_result.user_est > 5] = 5
    per_user_result["user_est"][per_user_result.user_est < 1] = 1
    for movie_path in per_movie_result_files:
        per_movie_result = pd.merge(ratings_test, pd.read_csv(movie_path))
        per_movie_result["movie_est"][per_movie_result.movie_est > 5] = 5
        per_movie_result["movie_est"][per_movie_result.movie_est < 1] = 1
        combi_result = pd.merge(per_movie_result[["rating", "rating_id", 'movie_est', 'movie_est_var']],
                                per_user_result[["rating_id", 'user_est', 'user_est_var']],
                                on="rating_id")
        combi_result["movie_est"] = combi_result["movie_est"].fillna(combi_result.user_est)
        combi_result["movie_est_var"] = combi_result["movie_est_var"].fillna(combi_result.user_est_var)
        combi_result["user_est"] = combi_result["user_est"].fillna(combi_result.movie_est)
        combi_result["user_est_var"] = combi_result["user_est_var"].fillna(combi_result.movie_est_var)
        scores = get_validation_score_combine_var(combi_result["rating"],
                                                  combi_result["user_est"],
                                                  combi_result["movie_est"],
                                                  combi_result["user_est_var"],
                                                  combi_result["movie_est_var"])
        scores["usr_m"] = "usr_" + os.path.basename(user_path)[9:-9]
        scores["mov_m"] = "mov_" + os.path.basename(movie_path)[10:-9]
        scores["combi_type"] = "var_t1"

        scores2 = get_validation_score_combine_simple(combi_result["rating"],
                                                      combi_result["user_est"],
                                                      combi_result["movie_est"])
        scores2["usr_m"] = "usr_" + os.path.basename(user_path)[9:-9]
        scores2["mov_m"] = "mov_" + os.path.basename(movie_path)[10:-9]
        scores2["combi_type"] = "simple_avg"

        per_combi_scores.append(scores)
        per_combi_scores.append(scores2)

per_combi_scores = pd.DataFrame(per_combi_scores)

# ------ #
# Scores #
# ------ #
print(per_movie_scores.sort_values("explained_var_score", ascending=False))
print()
print(per_user_scores.sort_values("explained_var_score", ascending=False))
print()
print(per_combi_scores.sort_values("explained_var_score", ascending=False))

"""

                      explained_var_score  mean_abs_err  mean_sqr_err  median_abs_err  r2_score  est_avg_var
mov_user_easy_rbf                0.230835      0.782339      0.957510          0.6848  0.230829     1.115123
mov_user_easy_linear             0.230824      0.782309      0.957524          0.6848  0.230818     1.115126
mov_user_easy_cos                0.230784      0.782343      0.957573          0.6848  0.230778     1.115117

                           explained_var_score  mean_abs_err  mean_sqr_err  median_abs_err  r2_score
usr_movie_easy_rbf                    0.166734      0.812776      1.037867         0.69240  0.166331
usr_word2vec_genre_rbf                0.154457      0.821043      1.052652         0.70180  0.154455
usr_genre_prob_rbf                    0.153551      0.820389      1.053942         0.70150  0.153419
usr_movie_easy_linear                 0.148726      0.822206      1.060788         0.70770  0.147920
usr_word2vec_genre_linear             0.143160      0.827058      1.066716         0.70655  0.143158
usr_word2vec_movie_rbf                0.142473      0.827827      1.067599         0.70590  0.142449
usr_movie_normal_rbf                  0.142419      0.827347      1.067637         0.70670  0.142419

    explained_var_score  mean_abs_err  mean_sqr_err  median_abs_err  r2_score                      usr_m                 mov_m  combi_type
16             0.275684      0.761490      0.902272        0.664824  0.275248         usr_movie_easy_rbf     mov_user_easy_rbf      var_t1
14             0.275639      0.761506      0.902328        0.664878  0.275203         usr_movie_easy_rbf  mov_user_easy_linear      var_t1
12             0.275628      0.761512      0.902341        0.664854  0.275192         usr_movie_easy_rbf     mov_user_easy_cos      var_t1
34             0.275358      0.763422      0.902313        0.663551  0.275215     usr_word2vec_genre_rbf     mov_user_easy_rbf      var_t1
32             0.275310      0.763443      0.902373        0.663580  0.275167     usr_word2vec_genre_rbf  mov_user_easy_linear      var_t1
30             0.275304      0.763450      0.902381        0.663604  0.275161     usr_word2vec_genre_rbf     mov_user_easy_cos      var_t1
17             0.274353      0.763966      0.903483        0.665350  0.274276         usr_movie_easy_rbf     mov_user_easy_rbf  simple_avg
15             0.274318      0.763974      0.903526        0.665400  0.274241         usr_movie_easy_rbf  mov_user_easy_linear  simple_avg
13             0.274300      0.763985      0.903550        0.665350  0.274222         usr_movie_easy_rbf     mov_user_easy_cos  simple_avg
35             0.273312      0.766590      0.904686        0.664350  0.273309     usr_word2vec_genre_rbf     mov_user_easy_rbf  simple_avg
33             0.273277      0.766603      0.904730        0.664300  0.273274     usr_word2vec_genre_rbf  mov_user_easy_linear  simple_avg
31             0.273260      0.766616      0.904751        0.664350  0.273257     usr_word2vec_genre_rbf     mov_user_easy_cos  simple_avg
4              0.272764      0.764061      0.905737        0.664075  0.272465         usr_genre_prob_rbf     mov_user_easy_rbf      var_t1
2              0.272718      0.764083      0.905794        0.664193  0.272419         usr_genre_prob_rbf  mov_user_easy_linear      var_t1
0              0.272708      0.764088      0.905807        0.664194  0.272408         usr_genre_prob_rbf     mov_user_easy_cos      var_t1
5              0.270701      0.767069      0.907958        0.664750  0.270680         usr_genre_prob_rbf     mov_user_easy_rbf  simple_avg
3              0.270666      0.767082      0.908003        0.664750  0.270645         usr_genre_prob_rbf  mov_user_easy_linear  simple_avg
1              0.270647      0.767092      0.908026        0.664750  0.270626         usr_genre_prob_rbf     mov_user_easy_cos  simple_avg
22             0.269348      0.766795      0.909818        0.664184  0.269187       usr_movie_normal_rbf     mov_user_easy_rbf      var_t1
20             0.269303      0.766818      0.909874        0.664283  0.269142       usr_movie_normal_rbf  mov_user_easy_linear      var_t1
18             0.269292      0.766826      0.909889        0.664302  0.269130       usr_movie_normal_rbf     mov_user_easy_cos      var_t1
40             0.269132      0.767212      0.910031        0.663567  0.269016     usr_word2vec_movie_rbf     mov_user_easy_rbf      var_t1
38             0.269086      0.767235      0.910090        0.663592  0.268969     usr_word2vec_movie_rbf  mov_user_easy_linear      var_t1
36             0.269075      0.767242      0.910102        0.663628  0.268958     usr_word2vec_movie_rbf     mov_user_easy_cos      var_t1
28             0.269050      0.767039      0.910238        0.663485  0.268850  usr_word2vec_genre_linear     mov_user_easy_rbf      var_t1
26             0.269003      0.767063      0.910297        0.663500  0.268802  usr_word2vec_genre_linear  mov_user_easy_linear      var_t1
10             0.269000      0.765768      0.910927        0.665450  0.268296      usr_movie_easy_linear     mov_user_easy_rbf      var_t1
24             0.268994      0.767069      0.910308        0.663546  0.268793  usr_word2vec_genre_linear     mov_user_easy_cos      var_t1
8              0.268953      0.765790      0.910986        0.665487  0.268249      usr_movie_easy_linear  mov_user_easy_linear      var_t1
6              0.268943      0.765797      0.910999        0.665500  0.268238      usr_movie_easy_linear     mov_user_easy_cos      var_t1
11             0.267083      0.768696      0.912647        0.664950  0.266915      usr_movie_easy_linear     mov_user_easy_rbf  simple_avg
9              0.267046      0.768710      0.912693        0.664950  0.266877      usr_movie_easy_linear  mov_user_easy_linear  simple_avg
7              0.267029      0.768721      0.912715        0.664950  0.266860      usr_movie_easy_linear     mov_user_easy_cos  simple_avg
41             0.266748      0.770661      0.912871        0.664400  0.266734     usr_word2vec_movie_rbf     mov_user_easy_rbf  simple_avg
39             0.266711      0.770676      0.912917        0.664350  0.266697     usr_word2vec_movie_rbf  mov_user_easy_linear  simple_avg
37             0.266693      0.770688      0.912939        0.664400  0.266680     usr_word2vec_movie_rbf     mov_user_easy_cos  simple_avg
29             0.266653      0.770424      0.912973        0.664350  0.266653  usr_word2vec_genre_linear     mov_user_easy_rbf  simple_avg
27             0.266616      0.770440      0.913019        0.664350  0.266616  usr_word2vec_genre_linear  mov_user_easy_linear  simple_avg
25             0.266598      0.770451      0.913041        0.664350  0.266598  usr_word2vec_genre_linear     mov_user_easy_cos  simple_avg
23             0.266539      0.770432      0.913116        0.664750  0.266537       usr_movie_normal_rbf     mov_user_easy_rbf  simple_avg
21             0.266503      0.770446      0.913161        0.664750  0.266502       usr_movie_normal_rbf  mov_user_easy_linear  simple_avg
19             0.266485      0.770458      0.913183        0.664750  0.266484       usr_movie_normal_rbf     mov_user_easy_cos  simple_avg

"""
