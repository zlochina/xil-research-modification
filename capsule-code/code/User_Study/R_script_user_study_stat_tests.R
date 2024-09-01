library (car)
library (stats)
library (psych) 
library (Hmisc)
library (TeachingDemos)
library(readxl)
library (pwr)

### DATEN LESEN

Gesamtdatensatz_R <- read_excel("data/user_study/Gesamtdatensatz_R_subset.xlsx", 
                                       col_types = c("numeric", "text", "numeric", 
                                                     "text", "numeric", "numeric", 
                                                     "numeric", "text", "numeric", "numeric", 
                                                     "numeric", "numeric", "numeric", "numeric", 
                                                     "numeric", "numeric", "numeric", "numeric", "numeric", 
                                                     "numeric", "numeric", "numeric", "numeric", "numeric", 
                                                     "numeric", "numeric", "numeric", "numeric", "numeric", 
                                                     "numeric", "numeric", "numeric", "numeric", 
                                                     "numeric", "numeric", "numeric"), na = "NA")
Gesamtdaten_R_Vertrauen <- read_excel("data/user_study/Gesamtdaten_R_Vertrauen_subset.xlsx", 
                                      col_types = c("numeric", "text", "text", 
                                                    "numeric", "numeric"), na="NA")
### DATEN VORARBEITEN 


### Subsets bilden
# Subsets f�r die Versuchsbedingungen (Test Conditions (TC))
VB1 <- subset(Gesamtdatensatz_R, Gesamtdatensatz_R$VB=="1")
VB2 <- subset(Gesamtdatensatz_R, Gesamtdatensatz_R$VB=="2")
VB4 <- subset(Gesamtdatensatz_R, Gesamtdatensatz_R$VB=="4")

# Subsets f�r die Vertrauensfragen pro Versuchsbedingung (VB)
VertrauensDatenVB1 <- subset(Gesamtdaten_R_Vertrauen, Gesamtdaten_R_Vertrauen$VB=="1")
VertrauensDatenVB2 <- subset(Gesamtdaten_R_Vertrauen, Gesamtdaten_R_Vertrauen$VB=="2")
VertrauensDatenVB4 <- subset(Gesamtdaten_R_Vertrauen, Gesamtdaten_R_Vertrauen$VB=="4")

# Subsets f�r die Bl�cke (jeweils 10 images)
Block1 <- subset (Gesamtdaten_R_Vertrauen, Gesamtdaten_R_Vertrauen$Block=="1")
Block2 <- subset (Gesamtdaten_R_Vertrauen, Gesamtdaten_R_Vertrauen$Block=="2")
Block3 <- subset (Gesamtdaten_R_Vertrauen, Gesamtdaten_R_Vertrauen$Block=="3")

# TiA als Daten zusammenf�gen 
TiA <- cbind(Gesamtdatensatz_R$TiAQ1,Gesamtdatensatz_R$TiAQ2, Gesamtdatensatz_R$TiAQ3, Gesamtdatensatz_R$InvertTiAQ5, Gesamtdatensatz_R$TiAQ6, Gesamtdatensatz_R$TiAQ7, Gesamtdatensatz_R$TiAQ9, Gesamtdatensatz_R$InvertTiAQ10, Gesamtdatensatz_R$TiAQ11, Gesamtdatensatz_R$TiAQ12, Gesamtdatensatz_R$TiAQ13, Gesamtdatensatz_R$TiAQ14, Gesamtdatensatz_R$InvertTiAQ15, Gesamtdatensatz_R$InvertTiAQ16, Gesamtdatensatz_R$TiAQ17, Gesamtdatensatz_R$TiAQ18, Gesamtdatensatz_R$TiAQ19)

### HYPOTHESEN TESTUNG 

## Vor den Berechnungen f�r die Hypothesen, Bonferroni-Holm korrifierte Alpha Werte berechnen 
0.05/4
# p-wert = 0.0125
0.05/3
# p-wert = 0.01666667
0.05/2
# p-wert = 0.025
0.05/1
# p-wert = 0.05

#txtStart("Ergebnisse der Anovas")

#Start mit der MANOVA  
cat("------------------------------------------------------\n")
cat("Leven Test for familiarity: \n")
LeveneTestManova <- leveneTest (Gesamtdatensatz_R$TiAQ_Score ~ Gesamtdatensatz_R$VB*Gesamtdatensatz_R$Erfahrung_KI)
print (LeveneTestManova)
cat("------------------------------------------------------\n")

# Manova für Vertrautheit
cat("------------------------------------------------------\n")
cat("MANOVA for familiarity: \n")
Manova <- aov(TiAQ_Score ~ VB*Erfahrung_KI, data=Gesamtdatensatz_R)
summary (Manova)
model.tables(Manova, "means")
describe.by(Gesamtdatensatz_R, Gesamtdatensatz_R$Erfahrung_KI, na.rm=TRUE)
cat("------------------------------------------------------\n")

cat("------------------------------------------------------\n")
cat("Post-hoc Pairwise Test between TCs: \n")
pairwise.t.test(Gesamtdatensatz_R$TiAQ_Score, Gesamtdatensatz_R$VB)
cat("------------------------------------------------------\n")

#ANOVA Vertrauen in Regellernen pro VB also 4x) 
#Anova: Gibt es Unterschiede im mittelten Vertrauen in die korrekte Regelerkennung der KI in den unterschieldichen Bl�cken innerhalb der VB? 
#VB1
cat("------------------------------------------------------\n")
cat("Levene for trust in rule learning TC1: \n")
leveneTestAnova5 <- leveneTest(VertrauensDatenVB1$Regel_Vertrauenswert_Lokal ~ VertrauensDatenVB1$Block)
print(leveneTestAnova5)
Anova5 <- aov(Regel_Vertrauenswert_Lokal ~ Block, data=VertrauensDatenVB1)
cat("ANOVA for trust in rule learning TC1: \n")
summary (Anova5)
model.tables(Anova5, "means")
TukeyHSD(aov(Regel_Vertrauenswert_Lokal ~ Block, data=VertrauensDatenVB1))
cat("------------------------------------------------------\n")

#VB2
cat("------------------------------------------------------\n")
cat("Levene for trust in rule learning TC2: \n")
leveneTestAnova6 <- leveneTest(VertrauensDatenVB2$Regel_Vertrauenswert_Lokal ~ VertrauensDatenVB2$Block)
print(leveneTestAnova6)
Anova6 <- aov(Regel_Vertrauenswert_Lokal ~ Block, data=VertrauensDatenVB2)
cat("ANOVA for trust in rule learning TC2: \n")
summary (Anova6)
model.tables(Anova6, "means")
TukeyHSD(aov(Regel_Vertrauenswert_Lokal ~ Block, data=VertrauensDatenVB2))
cat("------------------------------------------------------\n")

#VB4
cat("------------------------------------------------------\n")
cat("Levene for trust in rule learning TC3: \n")
leveneTestAnova8 <- leveneTest(VertrauensDatenVB4$Regel_Vertrauenswert_Lokal ~ VertrauensDatenVB4$Block)
print(leveneTestAnova8)
Anova8 <- aov(Regel_Vertrauenswert_Lokal ~ Block, data=VertrauensDatenVB4)
cat("ANOVA for trust in rule learning TC3: \n")
summary (Anova8)
model.tables(Anova8, "means")
TukeyHSD(aov(Regel_Vertrauenswert_Lokal ~ Block, data=VertrauensDatenVB4))
cat("------------------------------------------------------\n")






