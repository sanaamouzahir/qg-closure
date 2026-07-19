I want you to do an organizational thing for me. 



Go into the cluster, and do the following in : **PATH TO THE SGS ENSEMBLE**, for each ensemble member copy the DNS\_seismic video and put in in this directory : 

**SGS CLOSURE DIR:** in a subdirectory that follows the naming conevntion fpc\_fullfctname\_Remin\_Remax  or fpCape\_fullfctname\_Remin\_Remax

In each of this subfolders also copy the Pi\_FF4,PiFF6 directories and 8so we can see them, but with gaussian filter (not the other ones)





Paper fixes: 

1\. Abstract: dont mention the size of the stencil, say user defined

2\. We add R" and R' not just R3 

3\. If we match the truth completely yes there is no room for error,but we have shown that if we match the truth up to a certain order, then we also have the higher order diffusion of the scheme itself

4\. Forget about emulating the same scheme. We never do that anymore. It's always a heap linear multi step method matched to a high order RK method

5\. Introduction: 

&#x09;We need to put A LOT of references. Numerics papers, comparisons between Linear multi steps methods, comparison of linear multistep methods and RK methods, pseudo spectral methods, pararel methods, deffered correction, 

Machine learning: Neural time integrator with stage correction, neurvec, deep euler, RK4 steps, papers with stability estimates and generalizations



6\. Contribution: 

1\. A cheap, generalized O(1000) physics informed neural correction with Von Neumann Penalty that adds negligeable overhead but gives us orders of magnitutes improvement and huge stability improvements (we can now run at orders of magntiude higher dts)

2\. A detailed error analysis of the estimator 

3\. A detailed stability analysis of the neural augmented AB2CN2 scheme

4\. A general framework for Linear Multi-step Methods. 



7\. Table 1: FNN== learned part of the closure

&#x09;in 2.1: the physical setting follows \[16] (remove the Vallis)

&#x09;3.3: no need to keep mentioning the RK4 macthes the true trajectory. We will only mention it ONCE, when we describe the architechture and shwo the results 



&#x09;Remark 2 of proposition 3: can you elaborate more ? Also can we do the same comparison of the full LTE to the Padé schele ?

&#x09;Remark 3 of proposition 3: remove references to same scheme vs K fine entierely. 

&#x09;

8\. 4.2: When you efube the members; define thm propelry: Start with the full equation, then for each memeber, have a table with the column F where u put the full F formula,kf nu, Beta only

then proposition 4.21 : Start with the expression of the lTE as a power series. Then do Cauchy Hadamart and then do Rudin theorem (where you clearly show the bounds) in their analytical form. Then you can have another table where you show per member the Dt and the brackets. 

&#x20;For the Taylor microscale have the full derivations we did in : 

9\. Proposition 5: more detail on the deruvation i think we had justified the relationthip between dt and the Taylor microscale

Regarding the empirical C we found, have them in a table where you show the C per member and have a sentence that refers the reader to the table where we define the parameters of each member

Same for 4.2.3, have these numbers in a table

Proposition 6 remove the reference to Re25K. we will mention that later when we show the results and when we show that our thing doesnt work for Re25k and we will explain that it is specifically bc we are outside of the radius of convergence



proposition 7: this is not a proposition. Just put we the amplification expression and then have a per member table of the numbers (combining 4.2.4 and what u called prop 7)

Remove collorary 1 after prop 7



4.3: cost analysis: Only have AB2CN2, AB4CN2 ,RK4, RK4 at DT/K (no NN analysis yet). Have them in a table. Two cost colimns: Run time and storage

No mention of speedup yet or anything NN related yet 



5.1 wab2CN2+ tau=w(t+h) + O(h)^p where p is the order of the truncation. Not truth exactly



5.2: No. This would be true if we were FULLy matching the truth, but we only match it up to 4 order (up to 5th locally), so its almost like creating a sort of AB5CN5. Have a palceholder for our stability analysis plots. Where we add the apriori closures. FYI the stability regions are almost the same. 



Here we actualy have to think. Bc our splt is such that we are actually latchig the truth at 5th order ( or rather using the LTE wrt to the truth to match RK4 at very tiny h). SO we need a little bit more justification, I.e for h> 0 h^5\*\*(...)> 0 SO LTE (AB2CN2 vs RK4 fine h)\~\~ LTE(AB2CN2 vs truth)



5.3 analytical/ learned split should incorporate R4 as well

Specify that the analytical part are diagonal in a pseudo spectral code, else they are matrix multiplications. Explicitely remind what these linear operators are in Fourrier space vs in spacial domain. 



5.4. You now need to explicitely say we are matching RK4 AT dt=DT/K (i forgot which we kept fixed dt or K but u can find out). Therefore the LTE is > what you have with 1/K^2  since K>>>>1 we ignore it

Inference should have R4 term as well. PLus u have a typo it hsould be N not N^n



6.1 do not name CheapDerivClosure and in general no namings of our codes or whathever we use when we code. This soon will be an caademic paper dont go around using code names or variable names we use when we code. So far, no mention of S=7. Keep S throughout.

Remove the order clip part

do not specify 15 stencil. Say W-stencil keep W generic. Apply W.. to the S differentiable fields. No explicit mention of the parameters. Everything should be generic till the end

\# of pamaterts should be expressed as a function of d and S. No explicit numbers.

6.2.1 Again, no explicit numbers. Eveerything exoressed in terms of generic variables. When we show the results we'll have S= W=

Remove the numbers u computed. Well have them later when we show results. Add reference to appendix where we have all of the full derivations in 

error\_analysis\_shallow\_nn.tex. 

6.3 have all the math then give the numbers we found empirically. Not in the middle of the explanation/ derivation. 

Conditioned architechture: Again, all the explanation NO numbers or parameters. Then table with numbers and parameters

Remove the predited ceilings part



Rollout fine-tune : remove the annulus enstrophy part for now. Again dont specity M and the window size (call it window size w). 

For the architechture part, we will fully consolidate lateronce we have a final version. 



Prop 8 you need to add the R4 term too so eNN\*o(h^4) or some good way to formalize that

remove colloraly 4



Data: It's forced AND decaying turbulence

at Re,B, k, (grids 512 and 256) at different DT

Remove pipline safeguars

Have otpimization parameter in a table





