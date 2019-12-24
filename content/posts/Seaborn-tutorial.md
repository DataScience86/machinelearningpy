---
author: "Mohit Sharma"
title:  "A Step By Step Seaborn Tutorial"
date: "2019-12-06"
description: "A complete guide on how to use Seaborn library for data visualization in python. A tool that makes exploratory data analysis easy and efficient."
tags:
- Python
- Tutorial
- Seaborn
- Exploratory data analysis
- Plot
- Graph


categories: 
- Python
- Visualization

libraries:
- mathjax

featured_image: "postfeatureimg/python.png"
---

## Introduction To Seaborn
**Seaborn** is a python library which is built on top of `matplotlib` package. The package is also closely integrated with `pandas` data structure. `Seaborn` functions aim to make exploring and data understanding easy through visualization. The functions provided in seaborn can work on data frames and arrays. While building graphs the functions can internally perform statistical aggregations and generate informative  graphs. In this article, we will leanr how to draw different types of charts using seaborn library in Python.

## How to install seaborn library
To install `seaborn` package use `pip` or `conda` in Jupyter Notebook as given below:

``` Python
pip install seaborn
# install using anaconda command line
conda install seaborn
# install using anaconda command line
!pip install seaborn
# install from jupyter notebook/spyder 
```

Once installed use import the library using `import` function. If you want you can also give alias for easy calling of the functions from the module.

``` Python
import seaborn as sns
```

## How to choose color palettes in Seaborn
For any sortof visualization colors play an important role. Colors can help highlight data patterns if used effectively. Seanborn provides wide range of color palettes. It also makes it easy to select and use these palettes. Now let's take a quick look into how we can change the color style. We can set the style using `set()` method.

``` Python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
```
You can use `color_palette()` function to generate colors in seaborn. Many functions in seaborn internally give provision to mention color palettes through `palette` argument. The package also provides pre-defined names of color palettes or colormaps. Let's check out the default color palette. The seaborn provides six variations of the default color theme.

The choice of color theme/scheme is dependent on the nature of your data.  You can learn more on [types of color schemes](http://colorbrewer2.org/learnmore/schemes_full.html#qualitative) from the color brewer website.

### Seaborn Default color palette
We can use the follow code to check the default color palette.
``` Python
sns.palplot(sns.color_palette())
```
![Default color palette](/images/seaborn/default-palette.png)

You can also change the number of colors you need through the use of `n_colors` argument. To change the default color theme use `palette` argumnet as given below.
``` Python
sns.palplot(sns.color_palette(palette = "pastel", n_colors = 4))
```
![Default pastel palette](/images/seaborn/pastel-4.png)

### Seaborn Choosing colors for arbitrary number of categories
You can use `hls` or `hls_palette()` color space when you want to visualize categorical variables without emphazising on a specific category. 

``` Python
sns.palplot(sns.color_palette("hls", 5))
sns.palplot(sns.hls_palette(5))
```
![hls palette](/images/seaborn/hls.png)

### Seaborn choosing colors from color brew palette
You can use [color brew tool](http://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3) for getting visually pleasing color palettes. Below I ma providing some of th most popular color palettes.

``` Python
sns.palplot(sns.color_palette("Set2"))
sns.palplot(sns.color_palette("Paired"))
```

### Seaborn choosing sequential color palette
This type of color scheme is used for sequential data. A sequential data has a range where low value are relatively uniteresting compared to high values or vis a versa. Seaborn provides a great number of options for these palettes. They are mostly named after the dominant color.
``` Python
sns.palplot(sns.color_palette("Blues"))
sns.palplot(sns.light_palette("purple", reverse=True))
```
![sequence palette](/images/seaborn/sequence.png)

### Choosing custom color palette
`color_palette()` function makes it very easy to set your own colors as part of chart style. 

``` Python
custom = ["#E5C494", "#FFFFFF", "#318EFE", "#e74c3c", "#34495e"]
sns.palplot(sns.color_palette(custom))
```
![custom palette](/images/seaborn/custom.png)
### How to change the color palette in seaborn
Color schemes can be changed in seaborn using `set_palette()` function. Use the blow code to set your style. 
``` Python
import seaborn as sns
sns.set_palette("purple")
```

## How to change seaborn themes
The seaborn package provides five themes to work with: `darkgrid`, `whitegrid`, `dark`, `white`, and `ticks`. The choice of theme is a personal preference. By default, the seaborn works with `darkgrid` theme.

You can set a new theme by using `set_style()` functions as mentioned in the code below. Once set all the following charts will be as per the new theme

``` Python
import seaborn as sns
sns.set_style("whitegrid")
tips = sns.load_datasets("tips")
```
![whitegrid theme in seaborn](/images/seaborn/theme-1.png)

## Seaborn - histogram Python
You can use `distplot()` from `seaborn` to **plot a histogram**. By default, the function also fits a [kernel density estimate](https://en.wikipedia.org/wiki/Kernel_density_estimation) also represented by KDE. In the code below we will see how to draw the historam with `pestal` color pestal.

``` Python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

seaborn.set_color_codes(palette='Dark2_r')
# Setting a new color palette style

randnum = np.random.normal(size=1000)
sns.distplot(randnum)
```

![histogram in seaborn](/images/seaborn/histogram-1.png)

### How to change the bins in histogram
```Python
sns.distplot(randnum, bins=30, kde=False)
```
![histogram with bins in seaborn](/images/seaborn/histogram-2.png)

### How to plot only the distribution
```Python
sns.distplot(randnum, bins=30, hist=False)
```
![distribution plot in seaborn](/images/seaborn/histogram-3.png)

## Seaborn - scatter plot Python
**Scatter plots** are the mainstay of statistical analysis. They help us understandthe relationship between two continuous variables. To draw a scatter plot in seaborn you can use either `relplot()` or `scatterplot()` functions. The `replot()` function can also be used to draw a `lineplot()` and can also provide other functionalities like generating facets and all.

To draw a simple scatter plot use the below code.

``` Python
tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", data=tips)

sns.scatterplot(x="total_bill", y="tip", data=tips)
# Same plot using scatterplot()
```
![simple scatter plot in seaborn](/images/seaborn/scatterplot-1.png)
### How to draw scatter plot by group variable
While two dimensions can provide a good insight into the data, adding another dimension could help us dig deeper into the data patterns and can reveal really good insights. Question is how can be add the third dimension? Well a third dimension can be added in multiple ways:

1. Using colors
2. using marker style
3. Using different sizes

#### How to draw scatter plot by group using color
To represent the thrid variable(categorical) we can use `hue` argument. Let us see, between smokers and non-smoker who gave more tips in relationship to the total_bill.

``` Python
sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips)
sns.scatterplot(x="total_bill", y="tip", hue="smoker", data=tips)
```
![group by scatter plot in seaborn - using color](/images/seaborn/scatterplot-2.png)
#### How to create scatter plot by group variable using marker
To add the marker you can use `style` argument.
``` Python
sns.set_palette("Dark2")
sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker",   data=tips)
sns.scatterplot(x="total_bill", y="tip", hue="time", style="time", data=tips)
```
![group by scatter plot in seaborn - using marker](/images/seaborn/scatterplot-3.png)

> In the above examples, the third variable added is categorical. But if a numerical variables is passed in`hue` argumnet then the plot either selects sequential colors or uses size to distinguish the data points on the scatter plot. 

#### How to create scatter plot by group variable using size
``` Python
sns.relplot(x="total_bill", y="tip", hue="size", data=tips)
sns.scatterplot(x="total_bill", y="tip", hue="size", data=tips)
```
![group by scatter plot in seaborn - using size](/images/seaborn/scatterplot-4.png)

## Seaborn - line chart Pyhton
Mostly, we draw line charts when we are interested in knowing how the values of a specific variable changes over time. To draw line plot you can use `relplot()` functions with argumnet `kind = line()` or you can use `lineplot()` functions. In the coming examples we will be using the functions interchangibaly.

### Example #1 Seaborn default line Chart
For example, we will be draw a trend chart showing the **suicides/100k pop** trend from 1985 till 2016.

``` Python
suicide = pd.read_csv("master.csv")
sns.relplot(x="year", y="suicides_no", sort=False, kind="line", data=suicide);
```
![simple line plot in seaborn](/images/seaborn/lineplot-1.png)


### Example #2 Seaborn only line Chart no CI
By default, the `relplot()` for `kind = "line"` provides the confidence interval. The light blue region in the above chart represents the confidence intervals. If you do not want these then you need to pass `ci = None` as an argumnet. 

``` Python
sns.lineplot(x="year", y="suicides/100k pop", sort=True, ci = None, data=suicide)
```
![line plot without confidence intervals in seaborn](/images/seaborn/lineplot-2.png)

### Example #3 Seaborn mutiple lines plot
Just like `scatterplot()` the `lineplot()` is also flexible and can show upto three aditional variables using hue, style, and size elements. Let's check how the **suicides/100k pop** have changed per year for different genders.

``` Python
sns.lineplot(x="year", y="suicides/100k pop", sort=True, hue = "sex", data=suicide)
```
![multiple lines on a chart seaborn](/images/seaborn/lineplot-3.png)

It is evident from the above that suicides per 100k population is way much higher than the female. Although, both genders saw declining trend till 2013. However, 2015 saw a steep growth in suicide for males.

### Example #3 Seaborn Multiple line Chart 
Let's look at one more example by plotting the above parameter by age groups.

``` Python
sns.relplot(x="year", y="suicides/100k pop", sort=True, hue = "age",\
            ci = None, kind = "line" , data=suicide)
```
![line plot without confidence intervals in seaborn](/images/seaborn/lineplot-4.png)

So far we learned to visualize relationships between continuous varibales using `histograms`, `scatter plots` adn `line plots`. We learned how to use `hue`, `size` and `style` to understand these variables by the group variables.


## Seaborn plotting categorical variables

> **Seaborn** provides multiple different ways in which we can visualize relationships between categorical variables. You can draw categorical plots using `catplot()` function aling with `kind` argument or by using the below mentioned functions.

The different kinds of categorical plots are:  
1. **Distribution plots for categorical variables**
- `boxplot()`
- `boxenplot()`

2. **Point Estimate plots for categorical variables**
- `barplot()`
- `countplot()`
- `pointplot()`

3. **Scatter plots for categorical variables**
- `stripplot()`
- `swarmplot()`

We will discuss and learn how to draw all the above mentioned plots in `seaborn`. For all the seaborn examples we will be using suicide dataset.


## Seaborn - boxplot Python
A **boxplot** visualization helps us to understand the distribution of quantitative variable across different levels of a categorical variable. The boxplots are also widely used to **identify outliers** using a method which is based on inter-quartile range. 

### Example #1 - Verticle boxplot seaborn
Let's see what is the distribution of **gdp_per_capita ($)** variable looks like for different **sex** groups using the **boxplot**. 

``` Python
sns.boxplot(x="sex", y="gdp_per_capita ($)", data=suicide)
sns.catplot(x="sex", y="gdp_per_capita ($)", kind = "box", data=suicide)
```
![boxplot in seaborn](/images/seaborn/boxplot-1.png)

### Example #2 - Horizontal boxplot seaborn
In order to draw the horizontal boxplot we need to change the orientation of the axes usin `orient="h"` argument. Plus, you should swap the x and y arguments. So that y represents the categorical variable and x represents the continuous variable.

``` Python
sns.boxplot(y="sex", x="gdp_per_capita ($)", orient="h",data=suicide)
sns.catplot(y="sex", x="gdp_per_capita ($)", kind = "box", orient = "h", data=suicide)
```
![boxplot in seaborn](/images/seaborn/boxplot-2.png)

### Example #3 - Boxplot with 2nd categorical variable seaborn
If you wish to deep dive and learn how the ditribution of numerical variable changes by a group variable you can use `hue` argument, just like scatter plot.

``` Python
tips = sns.load_dataset("tips")
sns.boxplot(x="size", y="total_bill", hue="day", data=tips)
sns.catplot(x="size", y="total_bill", hue = "day" , kind = "box", data=tips)
```
![boxplot in seaborn](/images/seaborn/boxplot-3.png)

### Example #4 - Bonus - Genrating grids
The `catplot()` function allows you to conclude your visualization using facets. To generate grids or facets pass the categorical variable name in `col` argument. In case the number of levels for the categorical variable which is passed in `col` argument is high you can use `col_wrap` argument as shown below.

``` Python
sns.catplot(x="cut", y="price", col = "clarity",
            col_wrap = 4, data = diam, kind="box",
            height=4, aspect=.7)
```
![boxplot in seaborn](/images/seaborn/boxplot-4.png)

## Seaborn - boxenplot Python
A **boxenplot** is very similar to a **boxplot** as it used to visualize distribution using quantiles. But compared to boxplot the boxenplot gives more infomration about the shape of distribution especially at the tails.

**Let's see how some of the above drawn box plots will look when populated as boxenplot.**

### Example #1 - Verticle boxenplot seaborn

``` Python
sns.boxenplot(x="sex", y="gdp_per_capita ($)", data=suicide)
sns.catplot(x="sex", y="gdp_per_capita ($)", kind = "boxen", data=suicide)
```
![boxenplot in seaborn](/images/seaborn/boxenplot-1.png)

### Example #2 - Horizontal boxenplot seaborn

``` Python
sns.boxenplot(y="sex", x="gdp_per_capita ($)", 
              orient="h",data=suicide)
sns.catplot(y="sex", x="gdp_per_capita ($)", kind = "boxen", 
            orient = "h", data=suicide)
```
![horizontal boxenplot in seaborn](/images/seaborn/boxenplot-2.png)

### Example #3 - Boxenplot with 2nd categorical variable seaborn

``` Python
tips = sns.load_dataset("tips")
sns.boxenplot(x="size", y="total_bill", 
            hue="day", data=tips)
sns.catplot(x="size", y="total_bill", 
            hue = "day" , kind = "boxen", data=tips)
```
![boxenplot with grouped variable in seaborn](/images/seaborn/boxenplot-3.png)

## Seaborn - countplot Python
The **count plot** is also a dribution plot which is similar to hitorgrams. The only difference being that count plots are generated for categorical variable as compared to histograms which help visualize dritribution of quantitative variables.

### Example #1  Verticle count plot
For this example, lets see the distribution of categorical `age` variable from the **suicide** dataset. 

``` Python
sns.countplot(x="age", data=suicide)
sns.catplot(x="age", kind = "count", data=suicide)
```
![verticale count plot in seaborn](/images/seaborn/countplot-1.png)

The above chart sugests we have equal number of observations in all six age categories. 

### Example #2  Horizontal count plot
To draw a horizontal count plot you can mention the variable name in argment `y` instead of `x`. Let's check the distribution of generation variabe.

``` Python
sns.countplot(y="generation", data=suicide)
sns.catplot(y="generation", kind = "count", data=suicide)
```
![boxenplot with grouped variable in seaborn](/images/seaborn/countplot-2.png)

### Example #3  countplot by a categorical variables
We can use hue to look at the distribution of categorical variable by the grouped variable. Let's dig deeper into generation to check out how the distribution by male and female groups.

``` Python
sns.countplot(x="generation", hue ="sex", data=suicide)
sns.catplot(y="generation", hue = "sex", kind = "count", data=suicide)
```
![boxenplot with grouped variable in seaborn](/images/seaborn/countplot-3.png)

## Seaborn - barplot Python
In seaborn barplots are used to plot the central tendency estimates for the numerical variables along with the error bars. 

### Example #1 Vertical barplot
Let's check out the central tendency estimates of **gdp_per_capita ($)**  by the **generation** in suicide dataset.
``` Python
sns.barplot(x="generation", y ="gdp_per_capita ($)", data=suicide)
sns.catplot(y="generation", hue = "sex",kind = "bar", data=suicide)
```
![barplot seaborn](/images/seaborn/barplot-1.png)

### Example #2 barplot changing colors
To change the color of bars you can use `color`, `palette`, or `facecolor` like arguments.

**:Using colors:**
``` Python
sns.barplot(x="generation", y ="gdp_per_capita ($)",
            color = "green", data =suicide)
```
![barplot seaborn color example](/images/seaborn/barplot-1.png)

**:Using palette:**
``` Python
sns.barplot(x="generation", y ="gdp_per_capita ($)",
            palette = "Blues_d", data =suicide)
```
![barplot seaborn palette example](/images/seaborn/barplot-2.png)

**:Using facecolor:**
``` Python
sns.barplot(x="generation", y ="gdp_per_capita ($)",
            linewidth=4,facecolor = (1, 1, 1, 0),  
            errcolor=".5", edgecolor = ".5", data =suicide)
```
![barplot seaborn facecolor example](/images/seaborn/barplot-3.png)

### Example #3 barplot - Three dimensions

``` Python
sns.barplot(x="generation", y ="gdp_per_capita ($)",
            hue = "sex", data =suicide)
```
![barplot seaborn three dimention example](/images/seaborn/barplot-4.png)

## Seaborn - pointplot Python
**Point plots** are similar to bar plots as they also show the central tendency estimate along with the error bars. However, you may find points to be more useful for comparing levels of one or more than one categorical variable. 

### Examplt #1 point plot Python

``` Python
sns.pointplot(x="generation", y ="gdp_per_capita ($)", data =suicide)
```
![point plot seaborn](/images/seaborn/pointplot-1.png)


### Examplt #2 point plot by group variable Python
``` Python
tips = sns.load_dataset("tips")
sns.pointplot(x="size", y ="total_bill", 
              hue = "day",data =tips, dodge = True)
              
sns.catplot(x="size", y ="total_bill", 
              hue = "day", data =tips, 
            dodge = True, kind = "point")
```
![point plot seaborn](/images/seaborn/pointplot-2.png)

## Seaborn - stripplot Python
The `stripplots` can be used as substitutes for boxplots especially when you want to showcase all the observations as data points. 

### Example #1 Stripplot in Python

``` Python
import seaborn as sns
sns.set(style="darkgrid")
titanic = sns.load_dataset("titanic")
sns.stripplot(y=titanic.fare)
```
![stripplots seaborn](/images/seaborn/stripplot-1.png)

### Example #2 Stripplot group by categorical variable

``` Python
sns.stripplot(x = "class",y = "fare", data = titanic)
sns.catplot(x = "class",y = "fare", 
            data = titanic, kind = "strip")
```
![stripplots with categorical variable seaborn](/images/seaborn/stripplot-2.png)

### Example #3 Stripplot with jitters
**Jitters** can be really helpful in visualizing the distribution. To help you understand the importance we will be drawing two charts one with `jitter = False` as argument and one with `jitter = True`.

**Below is the plot with `jitter = False`**
``` Python
sns.stripplot(x = "class",y = "age", 
              data = titanic, jitter = False)
```
![stripplots without jitters seaborn](/images/seaborn/stripplot-3.png)  

**Below is the plot with `jitter = True`**
```Python
sns.catplot(x = "class",y = "age", 
            data = titanic, kind = "strip", jitter = True)
```
![stripplots with jitters seaborn](/images/seaborn/stripplot-4.png)

### Example #4 Stripplot with two categorical variables

``` Python
sns.stripplot(x = "class",y = "age",  hue = "alive",
              data = titanic, jitter = True)
sns.catplot(x = "class",y = "age", hue = "alive",
            data = titanic, kind = "strip", jitter = True)
```
![stripplots with two categorical variables seaborn](/images/seaborn/stripplot-5.png)

You can see that compared to third class more number of individuals servived in first class. 

### Example #5 Stripplot with differnt asthetics
In this example, we will see how change the following arguments
- shape of the markers: `marker = ` 
- use new palettes: `palette =` 
- size of the markers: `size =`
- opacity of markers: `alpha =`
- markeredge colors: `edgecolor =` 

``` Python
sns.stripplot(x = "class",y = "age",  hue = "alive", 
              palette="Set2", size=5, marker="D",
              edgecolor="red", alpha=.50, 
              data = titanic, jitter = True)
```
![stripplots with two categorical variables seaborn](/images/seaborn/stripplot-6.png)

## Seaborn - swarmplot Python
The plot is exactly same as `striplot` with only difference that unlike strip plot the points are adjusted such that they dont overlap. To built the plot you can use either `catplot()` function with `kind = swarm` argumnet or else you can use `swarmplot()` function.

### Example #1 swarmplot group by categorical variable

``` Python
sns.swarmplot(x = "class",y = "fare", data = titanic)
sns.catplot(x = "class",y = "fare", 
            data = titanic, kind = "swarm")
```
![swarmplot with categorical variable seaborn](/images/seaborn/swarmplot-1.png)

### Example #2 swarmplot with two categorical variable

``` Python
sns.swarmplot(x = "class",y = "age",  hue = "alive",
              data = titanic)
sns.catplot(x = "class",y = "age", hue = "alive",
            data = titanic, kind = "swarm")
```
![swarmplot with two categorical variable seaborn](/images/seaborn/swarmplot-2.png)

## Seaborn - FacetGrid Python
One of the most popuar ways to plot multiple plots by ensuring charts are still readable is through the use of **grids**. In seabborn library, we have  `FacetGrid()` function that provides this functionality. It can be used to draw upto three dimensions using `row`, `col` and `hue`. 

Let's see how we can plot fairs by class using titanic data. To generate grids we would be required to first create the facets.

``` Python
titanic = sns.load_dataset("titanic")
plot = sns.FacetGrid(data = titanic, col = "class")
```

![facetgrid multiple variables plot seaborn](/images/seaborn/facetgrid-1.png)

Now, that we have grids created we can map any chart using the `map` method. Let's add a scatter plot between `age` and `fare` per class.

``` Python
titanic = sns.load_dataset("titanic")
plot = sns.FacetGrid(data = titanic, col = "class")
plot.map(sns.scatterplot, "age", "fare")
```

![facetgrid mapping scatterplot seaborn](/images/seaborn/facetgrid-2.png)

Let's see how we could add the two categorical variables using `hue`.

``` Python
titanic = sns.load_dataset("titanic")
plot = sns.FacetGrid(data = titanic, col = "class", hue = "sex")
plot.map(sns.scatterplot, "age", "fare")
```

![facetgrid mapping scatterplot seaborn](/images/seaborn/facetgrid-3.png)

As we have too different coolor points on each plot. It is a good practice to add the `legend` for the ease of redability. To add legend we can use low level function called as `add_legend()`

``` Python
titanic = sns.load_dataset("titanic")
plot = sns.FacetGrid(data = titanic, col = "class", hue = "sex")
plot.map(sns.scatterplot, "age", "fare")
plot.add_legend()
```
![facetgrid mapping scatterplot by categorical variables with legend seaborn](/images/seaborn/facetgrid-4.png)
## Seaborn - pairwise plot Python

**Pairwise plot** is a very interesting plot. It represents a lot of information by plotting small subplots in a gird like arrangement. In each **gird** row and column is assigned to a different variable creating a pairwise bivariate relationship plot at the intersection. The diagonal plots represent univariate information, mostly related to ditribution. 

The `seaborn` package in Python provides two functions IE `PairGrid()` and `pairplot()` using which you can plot the pairwise relationships. Let's look at the two functions one by one. 

**For examples on pairwise relationship plots we will be using car crashes dataset from seaborn package**

### Seaborn - PairGrid Python Pairwise plot

The `PairGrid()` function is very similar to `FacetGrid()`. Just like `FacetGrid` you need to first initialize the grid and then useing `map` method generate the plots. 

``` Python
crashes = sns.load_dataset('car_crashes')
crashes.head()
```

![crashes table seaborn](/images/seaborn/pairwise-1.png)

``` Python
plot = sns.PairGrid(crashes)
plot.map(plt.scatter)
```
![PairGrid plot seaborn](/images/seaborn/pairwise-2.png)

The pariwise matrix shown above can be divided into following parts and subparts:

- Diagonal charts
- Off diagonal charts
  - upper triangle charts
  - lower triangle charts
  
> You can refer to these parts and change the chart type as per your requirment.

### Example #1 Changing diagonal charts
Updating diagnoal charts to histogram from scatter plot.
``` Python
plot = sns.PairGrid(crashes.iloc[:, 3:8])
plot.map_diag(plt.hist)
plot.map_offdiag(plt.scatter)
```
![PairGrid plot seaborn](/images/seaborn/pairwise-3.png)

### Example #2 Changing off-diagonal charts
Changing upper triangle charts to kdeplot while keeping the diagonal charts as histograms.

``` Python
plot = sns.PairGrid(crashes.iloc[:, 3:7])
plot.map_diag(plt.hist)
plot.map_uper(sns.kdeplot)
plot.map_lower(plt.scatter)
```
![PairGrid plot changed off-daigonal charts seaborn](/images/seaborn/pairwise-4.png)

### Example #3 Adding Categorical variable
In this example, we will be using `mpg` dataset. Lets' see how the pairwise plot looks like by the `cylinder` variable. And now, because we are trying to understand the distribution or relationship between levels of a categorical variable we will be adding a legend as well.

``` Python
plot = sns.PairGrid(mpg, vars = ['displacement', 'horsepower', 'weight'], hue = "cylinders")
plot.map_diag(plt.hist)
plot.map_offdiag(plt.scatter)
plot.add_legend()
```
![PairGrid plot with categorical variable seaborn](/images/seaborn/pairwise-5.png)

### Seaborn - pairplot Python
The other function which you can use to create a pariwise plot is `pairplot()`. The function is less flexible but faster in comparison to `PairGrid()`. Let's quickly see how we can recreate the some of the above plot. You can use `pairplot()` for quick analysis but for deeper and more customizations `PairGrid()` function is a much better choice. 

Plotting a pairwise plot with cylinder as categorical variable.

``` Python
sns.pairplot(mpg, vars = ['displacement', 'horsepower', 'weight'], hue="cylinders")
```
![Pairplot with categorical variable seaborn](/images/seaborn/pairwise-6.png)

Changing diagonal plots to histograms.

``` Python
sns.pairplot(mpg, vars = ['displacement', 'horsepower', 'weight'], hue="cylinders")
```
![Pairplot with categorical variable seaborn](/images/seaborn/pairwise-7.png)


## Seaborn - jointplot Python
The `jointplot()` function is used when you wish to visualise the bivariate dsitribution of two variables in the same chart. The function creates a scatter plot between the two variables with histograms of each one of them on right and top axes.

``` Python
mpg = sns.load_dataset('mpg')
mpg["cylinders"] = mpg["cylinders"].astype('category')
sns.jointplot(x="mpg", y="weight", data=mpg)
```
![jointplot seaborn Python](/images/seaborn/jointplot-1.png)

You can also change the type of the plot using the `kind = ` argument. Let's see what happens when we change kind to **kde(Kernel Density Estimate)**.

``` Python
sns.jointplot(x="mpg", y="weight", data=mpg, kind = 'kde')
```
![jointplot with kind kde seaborn Python](/images/seaborn/jointplot-2.png)


## Seaborn - Visualizing with facets Python
You can visualize the multivariate relationships using facets. Almost all the functions which we have discussed so far provide an argument called `col = ` which can be used to generate facets.To generate grids or facets pass the categorical variable name in `col` argument. In case the number of levels for the categorical variable which is passed in `col` argument is high you can use `col_wrap` argument as shown below.

``` Python
diam = sns.load_dataset('diamonds')
sns.catplot(x="cut", y="price", col = "clarity",
            col_wrap = 4, data = diam, kind="box",
            height=4, aspect=.7)
```
![Visualizing with facets Python in seaborn](/images/seaborn/boxplot-4.png)

## Seaborn - regression line Python
The two functions which you can use to add the linear regression model line to the scatter plot are `regplot()` and `lmplot()`. The outputs of the functions are identical apart from the slight changes in the plot size. There is reason why this happens but is beyond the scope of this tutorial.

Let's add the regression line to the scatter plot of `mpg` Vs. `weight`.

``` Python
sns.regplot(x="mpg", y="weight", data=mpg)
```

![regression line to the scatter Python in seaborn](/images/seaborn/regplot-1.png)

## Seaborn - important style parameter
**Seaborn provides lots of options for you to customise your charts**. You can use `axes_style()` and`set_style()` functions to overide the default setting. To change a parameter you need to pass the new parameters as part of dictionary. To understand what all parameters you can change, call the function `axes_style()` without any  argument as given below.

``` Python
import seaborn as sns
sns.axes_style()
```

``` Markdown
{'axes.facecolor': '#EAEAF2',
 'axes.edgecolor': 'white',
 'axes.grid': True,
 'axes.axisbelow': True,
 'axes.labelcolor': '.15',
 'figure.facecolor': (1, 1, 1, 0),
 'grid.color': 'white',
 'grid.linestyle': '-',
 'text.color': '.15',
 'xtick.color': '.15',
 'ytick.color': '.15',
 'xtick.direction': 'out',
 'ytick.direction': 'out',
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'image.cmap': 'rocket',
 'font.family': ['sans-serif'],
 'font.sans-serif': ['Arial',
  'DejaVu Sans',
  'Liberation Sans',
  'Bitstream Vera Sans',
  'sans-serif'],
 'patch.force_edgecolor': True,
 'xtick.bottom': False,
 'xtick.top': False,
 'ytick.left': False,
 'ytick.right': False,
 'axes.spines.left': True,
 'axes.spines.bottom': True,
 'axes.spines.right': True,
 'axes.spines.top': True}
```

> The package also provides a high-level functions `set()` which can be used to change the plot parameters. However, the function takes a dictinary of matplotlib parameters as input.

Please leave comments, if
1. You find anything incorrect.
2. You want to add more information to the topic.
3. You wish to add another example to the topic.
4. You need more details in regards to a specific section.
5. You are unable to execute an example code.


