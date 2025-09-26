# Reflection: KNN Recommender on Potions.gg

## What data are you using for your recommendation and why?
I used the **subscriptions** file to find the publisher with the most subscribers (`wn32`). I then used the **content views** file to see what items those subscribers had watched. I built a table (user–item matrix) that marks a `1` if a user watched an item and `0` if they did not. I chose this because it is simple and avoids problems with comparing watch times.

---

## Which adventurers does your recommender serve well? Why?
The recommender works well for users who have watched many items that overlap with other users. For example, adventurer **`9w9y`** got both of their hidden items recommended (recall@2 = 1.0). The model could find patterns because this user had more common activity.

---

## Which adventurers does your recommender not serve well? Why?
It does not work well for users who watch fewer or less common items. For example, adventurers **`6dnb`** and **`3271`** got recall@2 = 0.0. The items hidden for them were not similar to their other watched items, so the model could not predict them.

---

## Why did you pick those three adventurers from the publisher you chose?
I picked three of the most active users from publisher `wn32`. I chose them because they had enough history to test recommendations and all came from the same publisher as required.

---

## Why do you believe your recommender chose the content it did for those adventurers?
The model looks at the items a user has already watched. It finds items that are similar (by cosine similarity) and then recommends the top two that the user has not seen yet. For `9w9y`, the hidden items were very similar to what they had already watched, so the system got them right. For `6dnb` and `3271`, the hidden items were not very similar, so the system gave other popular items instead.

---

## Evaluation Summary
- Method: For each user, I hid 2 items, recomputed recommendations, and checked recall@2.  
- Results: `9w9y` = 1.0, `6dnb` = 0.0, `3271` = 0.0.  
- Average recall@2 = 0.333.  
- This means the system worked well for one active user but not for the others, which shows the limits of a simple KNN recommender.
