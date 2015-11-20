# Conformal Prediction in Spark

Increasing size of datasets is challenging for machine learning, and Big Data frameworks, such as Apache Spark, have shown promise for facilitating model building on distributed resources. Conformal prediction is a mathematical framework that allows to assign valid confidence levels to object-specific predictions. This contrasts to current best-practices where the overall confidence level for predictions on unseen objects is estimated basing on previous performance, assuming exchangeability. This repository contains a Spark-based distributed implementation of conformal prediction, which introduces valid confidence estimation in predictive modeling for Big Data analytics.

## Introduction to Conformal Prediction

Conformal prediction is a mathematical framework that allows to assign valid confidence levels to object-specific predictions. This contrasts to current best-practices (e.g. cross-validation) where the overall confidence level for predictions on unseen objects is estimated basing on previous performance, assuming exchangeability. Given a user-specified significance level 𝜺, instead of producing a single prediction, a conformal predictor outputs a prediction set of outcomes. The major advantage of this approach is that, within the mathematical framework, there is proof that the true outcome for an unseen object will be in the prediction set with probability 1 - 𝜺.

### Nonconformity measure
The nonconformity measure is the core of the the Conformal Predictor theory. It is a user-defined function which assigns a strangeness measure to new examples, with respect to the examples that were used in order to train a machine learning model. 

## Getting started 
First, you need to setup a Spark project with maven, this tutorial is a good starting point:
[www.youtube.com/watch?v=aB4-RD_MMf0](www.youtube.com/watch?v=aB4-RD_MMf0).

Then, add the following entries into your pom.xml file: 

	<repositories>
		...
		<repository>
			<id>pele.farmbio.uu.se</id>
			<url>http://pele.farmbio.uu.se/artifactory/libs-snapshot</url>
		</repository>
		...
	</repositories>

	<dependencies>
	...
		<groupId>se.uu.farmbio</groupId>
			<artifactId>cp</artifactId>
			<version>0.0.1-SNAPSHOT</version>
		</dependency>
	...
	</dependencies>

## Usage
Please refer to the examples directory for usage.
