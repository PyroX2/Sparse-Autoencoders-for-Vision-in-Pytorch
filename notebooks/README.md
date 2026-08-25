# Opis na co reagują poszczególne neurony SAE
0. -
1. -
2. -
3. -
4. -
5. -
6. -
7. Zawsze na 6
8. -
9. -
10. -
11. -
12. -
13. -
14. -
15. -
16. -
17. -
18. na 7 albo fioletowe 0
19. -
20. -
21. -
22. Dla każdego zbioru co innego
23. -
24. -
25. -
26. -
27. -
28. -
29. -
30. -
31. -
32. białe 2 albo jaskrawy zielony (do usunięcia)
33. Zawsze na 3
34. Na 5 i może 0?
35. Na czerwone liczby (do usunięcia)
36. Białe i zielone jedynki i zielone 8 (chyba do usunięcia)
37. -
38. -
39. Białe 4 i blado żółte (do usunięcia)
40. -
41. -
42. -
43. na 9 albo niebieskie 7
44. Na 5 i na jasnoniebieskie (nie wiem)
45. -
46. -
47. -
48. -
49. Na dwójki i chyba blady żółty (nie wiem)
50. -
51. -
52. -
53. -
54. Na 9 i blado żółte 1
55. -
56. -
57. -
58. -
59. -
60. -
61. -
62. -
63. -
64. -
65. Na 3
66. -
67. -
68. -
69. -
70. -
71. -
72. -
73. -
74. -
75. -
76. -
77. -
78. -
79. Tylko na ciemnoniebieski (do wyrzucenia)
80. -
81. -
82. -
83. -
84. Trochę dziwne, chyba trochę na różowe (nie wiem)
85. -
86. W każdym datasecie coś innego
87. -
88. -
89. -
90. -
91. -
92. -
93. Na 4
94. -
95. -
96. -
97. -
98. -
99. -
100. -
101. Na białe 5/6 i różowe (do wyrzucenia raczej)
102. -
103. -
104. Raczej na 3 i czerwone 9
105. -
106. -
107. -
108. -
109. -
110. -
111. Tylko jaskrawy zielony (do wyrzucenia)
112. -
113. -
114. W każdym datasecie coś innego
115. -
116. Typowo na 0
117. -
118. Na 2
119. Na kolor czerwony (do wyrzucenia)
120. -
121. -
122. -
123. -
124. -
125. -
126. -
127. -
128. -
129. -
130. -
131. -
132. -
133. -
134. -
135. -
136. -
137. -
138. -
139. -
140. -
141. Na 6
142. -
143. -
144. -
145. Na 0 i 5
146. -
147. -
148. -
149. -
150. -
151. Głównie na 8 i czasem na blado żółte 1
152. -
153. -
154. -
155. -
156. -
157. -
158. -
159. -
160. -
161. -
162. -
163. -
164. -
165. -
166. -
167. -
168. Na 6
169. -
170. -
171. Na czerwony (do wyrzucenia)
172. -
173. -
174. -
175. -
176. -
177. -
178. Na 7 albo 9
179. -
180. -
181. -
182. Na ciemnoniebieski (do wyrzucenia)
183. -
184. -
185. -
186. -
187. Na 6 i różowy (nie wiem, chyba zostawić)
188. -
189. -
190. -
191. Na liczby podobne do 7
192. -
193. -
194. -
195. -
196. -
197. -
198. Na brązowe albo blado żółte (do wyrzucenia)
199. -
200. Na 7 i czerwony (nie wiem)
201. -
202. -
203. -
204. Głównie na różowy (do wyrzucenia)
205. -
206. -
207. -
208. -
209. -
210. -
211. -
212. -
213. -
214. -
215. -
216. -
217. -
218. Na kolorowe 6
219. Na dwójki i jasnoniebieskie 4
220. -
221. Na różowy (do wyrzucenia)
222. Na różowy i 2 (raczej do wyrzucenia)
223. -
224. -
225. Na 2 i jasnoniebieskie 5
226. -
227. -
228. -
229. -
230. -
231. -
232. -
233. -
234. Na 7 i brązowy (raczej do wyrzucenia)
235. -
236. Na czerwony (do wyrzucenia)
237. -
238. -
239. Na jedynko-podobne
240. -
241. -
242. -
243. -
244. -
245. -
246. W każdym coś innego
247. -
248. -
249. -
250. -
251. -
252. -
253. -
254. -
255. -
256. -


# Results
### Removal of neurons [31, 35, 38, 110, 197]:
```
=========== BASELINE ===========
Evaluating: 100%|██████████| 375/375 [00:01<00:00, 301.78it/s]
Evaluating: 100%|██████████| 375/375 [00:01<00:00, 275.68it/s]
Evaluating: 100%|██████████| 375/375 [00:01<00:00, 327.25it/s]
Performance on clean validation set:
Clean Val Loss: 0.015241414661208789, Accuracy: 0.8572604060173035, Precision: 0.8572604060173035, Recall: 0.8572604060173035, F1 Score: 0.8564618825912476, AUPRC: 0.9122222065925598, AUROC: 0.9766136407852173
Performance on fully colored validation set:
Fully Colored Val Loss: 0.00024403737605704616, Accuracy: 0.9996673464775085, Precision: 0.9996673464775085, Recall: 0.9996673464775085, F1 Score: 0.9996622800827026, AUPRC: 0.9998480677604675, AUROC: 0.999951183795929
Performance on validation set with colors flipped:
Colors flipped Val Loss: 0.15280315073331197, Accuracy: 0.3303631544113159, Precision: 0.3303631544113159, Recall: 0.3303631544113159, F1 Score: 0.3117218613624573, AUPRC: 0.3325870633125305, AUROC: 0.618381679058075
=========== TRIMMED ===========
Evaluating: 100%|██████████| 375/375 [00:01<00:00, 285.14it/s]
Evaluating: 100%|██████████| 375/375 [00:01<00:00, 286.06it/s]
Evaluating: 100%|██████████| 375/375 [00:01<00:00, 286.24it/s]Performance on clean validation set:
Clean Val Loss: 0.019227750840286415, Accuracy: 0.8230992555618286, Precision: 0.8230992555618286, Recall: 0.8230992555618286, F1 Score: 0.8231642842292786, AUPRC: 0.8915170431137085, AUROC: 0.9720733761787415
Performance on fully colored validation set:
Fully Colored Val Loss: 0.01341210949436451, Accuracy: 0.8883697390556335, Precision: 0.8883697390556335, Recall: 0.8883697390556335, F1 Score: 0.8447645306587219, AUPRC: 0.9200679063796997, AUROC: 0.9697611927986145
Performance on validation set with colors flipped:
Colors flipped Val Loss: 0.10340533173084258, Accuracy: 0.45948758721351624, Precision: 0.45948758721351624, Recall: 0.45948758721351624, F1 Score: 0.40913987159729004, AUPRC: 0.4378647804260254, AUROC: 0.694301962852478
```