Ce projet s'inscrit dans le cadre d'un stage d'encadrement des élèves de secondes en partenariat avec l'ENSEIRB-Matmeca et le CEA, le but étant de leur montrer les différentes filières de l'ingénerie. Dans ce cas, le but est de leur présenter la filière Mathématiques et Mécanique avec une expérience physique, puis une partie simulation numérique (construite ici).
Le code est donc interactif et les élèves peuvent jouer avec les paramètres afin de simuler différents scénarios.

Le code est divisé en deux parties, la première a pour but de simuler numériquement le phénomène de Mascaret se produisant dans une rivière. Plus particulièrement, on considérera une forme en entonnoir 
pour la rivière, avec un profil de bathymétrie réglable (linéaire, quadratique, constant). Cela permettra d'accentuer le ressaut et de mettre en avant le phénomène ondulatoire du Mascaret. Il est possible de mettre en place des obstacles mieux visualiser les ressauts. La visualisation se fait en 1D avec Matplotlib.
Le schéma numérique se base en partie sur celui d'article "Numerical Simulation of Tidal Bore Bono at Kampar River (JAFM, A. C. Bayu et al)"

La deuxième partie s'inspire de la première avec pour objectif de simuler le phénomène de Tsunami en 1D, on ne tient plus en compte de la largeur pour le schéma car on simule modélise désormais l'océan. Le scénario du tsunami se base en partie sur le séisme de 2011 du Japon.


