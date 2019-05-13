# Time Series project for predicting stock price using market news

suppressMessages(library(TSA))
suppressMessages(library(ggplot2))
suppressMessages(library(tseries)) 
suppressMessages(library(DataCombine))
suppressMessages(library(forecast))
suppressMessages(library(dplyr))
suppressMessages(library(Metrics))

library(reshape)
library(data.table)
library(formattable)
library(gridExtra)
library(funModeling)

library(psych)

library(glmulti) #Select the best linear regression model
library(lubridate) #Date Time Conversion
library(BBmisc) #Convert Rows to List
library(pROC) #ROC
library(DMwR) #Error Estimates
library(readr)
library(caret)
library(nlme)
library(magrittr)
library(stringr)
library(tidyverse)
library(randomForest)
library(e1071)
library(Metrics)







input_dir<-"F:/Projects/TimeSeries/EndGame/"

csv_files = list.files(input_dir, recursive = T, full.names = T)
csv_files = csv_files[grep('.csv', csv_files)]
csv_files

data = read.csv(csv_files[2], stringsAsFactors = F, nrows=100000)


# Function 1 : For ploting missing value
plot_missing <- function(data, title = NULL, ggtheme = theme_gray(), theme_config = list("legend.position" = c("bottom"))) {
  ## Declare variable first to pass R CMD check
  feature <- num_missing <- pct_missing <- group <- NULL
  ## Check if input is data.table
  is_data_table <- is.data.table(data)
  ## Detect input data class
  data_class <- class(data)
  ## Set data to data.table
  if (!is_data_table) data <- data.table(data)
  ## Extract missing value distribution
  missing_value <- data.table(
    "feature" = names(data),
    "num_missing" = sapply(data, function(x) {sum(is.na(x))})
  )
  missing_value[, feature := factor(feature, levels = feature[order(-rank(num_missing))])]
  missing_value[, pct_missing := num_missing / nrow(data)]
  missing_value[pct_missing < 0.05, group := "Good"]
  missing_value[pct_missing >= 0.05 & pct_missing < 0.4, group := "OK"]
  missing_value[pct_missing >= 0.4 & pct_missing < 0.8, group := "Bad"]
  missing_value[pct_missing >= 0.8, group := "Remove"][]
  ## Set data class back to original
  if (!is_data_table) class(missing_value) <- data_class
  ## Create ggplot object
  output <- ggplot(missing_value, aes_string(x = "feature", y = "num_missing", fill = "group")) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = paste0(round(100 * pct_missing, 2), "%"))) +
    scale_fill_manual("Group", values = c("Good" = "#1a9641", "OK" = "#a6d96a", "Bad" = "#fdae61", "Remove" = "#d7191c"), breaks = c("Good", "OK", "Bad", "Remove")) +
    scale_y_continuous(labels = comma) +
    coord_flip() +
    xlab("Features") + ylab("Number of missing rows") +
    ggtitle(title) +
    ggtheme + theme_linedraw()+
    do.call(theme, theme_config)
  ## Print plot
  print(output)
  ## Set return object
  return(invisible(missing_value))
}

# Function 2: For plotting histogram
plot_histogram <- function(data, title = NULL, ggtheme = theme_gray(), theme_config = list(), ...) {
  if (!is.data.table(data)) data <- data.table(data)
  ## Stop if no continuous features
  if (split_columns(data)$num_continuous == 0) stop("No Continuous Features")
  ## Get continuous features
  continuous <- split_columns(data)$continuous
  ## Get dimension
  n <- nrow(continuous)
  p <- ncol(continuous)
  ## Calculate number of pages
  pages <- ceiling(p / 16L)
  for (pg in seq.int(pages)) {
    ## Subset data by column
    subset_data <- continuous[, seq.int(16L * pg - 15L, min(p, 16L * pg)), with = FALSE]
    setnames(subset_data, make.names(names(subset_data)))
    n_col <- ifelse(ncol(subset_data) %% 4L, ncol(subset_data) %/% 4L + 1L, ncol(subset_data) %/% 4L)
    ## Create ggplot object
    plot <- lapply(
      seq_along(subset_data),
      function(j) {
        x <- na.omit(subset_data[, j, with = FALSE])
        ggplot(x, aes_string(x = names(x))) +
          geom_histogram(bins = 30L, ...,fill='#92b7ef') +
          scale_x_continuous(labels = comma) +
          scale_y_continuous(labels = comma) +
          ylab("Frequency") +
          ggtheme + theme_linedraw()+
          do.call(theme, theme_config)
      }
    )
    ## Print plot object
    if (pages > 1) {
      suppressWarnings(do.call(grid.arrange, c(plot, ncol = n_col, nrow = 4L, top = title, bottom = paste("Page", pg))))
    } else {
      suppressWarnings(do.call(grid.arrange, c(plot, top = title)))
    }
  }
}

# Function 3 : Getting missing values
.getAllMissing <- function(dt) {
  if (!is.data.table(dt)) dt <- data.table(dt)
  sapply(dt, function(x) {
    sum(is.na(x)) == length(x)
  })
}

# Function 4 : Spliting columns
split_columns <- function(data) {
  ## Check if input is data.table
  is_data_table <- is.data.table(data)
  ## Detect input data class
  data_class <- class(data)
  ## Set data to data.table
  if (!is_data_table) data <- data.table(data)
  ## Find indicies for continuous features
  all_missing_ind <- .getAllMissing(data)
  ind <- sapply(data[, which(!all_missing_ind), with = FALSE], is.numeric)
  ## Count number of discrete, continuous and all-missing features
  n_all_missing <- sum(all_missing_ind)
  n_continuous <- sum(ind)
  n_discrete <- ncol(data) - n_continuous - n_all_missing
  ## Create object for continuous features
  continuous <- data[, which(ind), with = FALSE]
  ## Create object for discrete features
  discrete <- data[, which(!ind), with = FALSE]
  ## Set data class back to original
  if (!is_data_table) class(discrete) <- class(continuous) <- data_class
  ## Set return object
  return(
    list(
      "discrete" = discrete,
      "continuous" = continuous,
      "num_discrete" = n_discrete,
      "num_continuous" = n_continuous,
      "num_all_missing" = n_all_missing
    )
  )
}

# Function 5 : plotting density plot for numerical variable 
plot_density <- function(data, title = NULL, ggtheme = theme_gray(), theme_config = list(), ...) {
  if (!is.data.table(data)) data <- data.table(data)
  ## Stop if no continuous features
  if (split_columns(data)$num_continuous == 0) stop("No Continuous Features")
  ## Get continuous features
  continuous <- split_columns(data)$continuous
  ## Get dimension
  n <- nrow(continuous)
  p <- ncol(continuous)
  ## Calculate number of pages
  pages <- ceiling(p / 16L)
  for (pg in seq.int(pages)) {
    ## Subset data by column
    subset_data <- continuous[, seq.int(16L * pg - 15L, min(p, 16L * pg)), with = FALSE]
    setnames(subset_data, make.names(names(subset_data)))
    n_col <- ifelse(ncol(subset_data) %% 4L, ncol(subset_data) %/% 4L + 1L, ncol(subset_data) %/% 4L)
    ## Create ggplot object
    plot <- lapply(
      seq_along(subset_data),
      function(j) {
        x <- na.omit(subset_data[, j, with = FALSE])
        ggplot(x, aes_string(x = names(x))) +
          geom_density(...,fill="#e2c5e5") +
          scale_x_continuous(labels = comma) +
          scale_y_continuous(labels = percent) +
          ylab("Density") +
          ggtheme + theme_linedraw()+
          do.call(theme, theme_config)
      }
    )
    ## Print plot object
    if (pages > 1) {
      suppressWarnings(do.call(grid.arrange, c(plot, ncol = n_col, nrow = 4L, top = title, bottom = paste("Page", pg))))
    } else {
      suppressWarnings(do.call(grid.arrange, c(plot, top = title)))
    }
  }
}


print(paste("The Dataset have",dim(data)[1],"data points and ", dim(data)[2], "Features."))

print("Let's see the head of data.")
print(head(data))

funModeling::df_status(data)

plot_missing(data)


getDataFrameWith50Categories <- function(df){
  factorDF <- dplyr::mutate_all(df, function(x) as.factor(x))
  features <- names(factorDF)
  for(feature in features){
    if(length(levels(factorDF[,feature]))>50){
      factorDF[feature] <- NULL
    }
    
  }
  factorDF         
}

categoricalData <- getDataFrameWith50Categories(data)

describe(categoricalData)

funModeling::freq(categoricalData)


getNumericalDF <- function(df){
  numericDF <- df
  features <- names(numericDF)
  for(feature in features){
    if(!is.numeric(df[,feature])){
      numericDF[feature] <- NULL
    }
  }
  numericDF
}

numericalData <- getNumericalDF(data) # which are num/int data type
numericalDataFeature <- names(numericalData) 
categoricalDataFeature <- names(categoricalData)
numericalData <- dplyr::select(numericalData, - dplyr::one_of(categoricalDataFeature)) # select that feature which are not categoricalDataFeature

funModeling::profiling_num(numericalData)


plot_histogram(numericalData)


plot_density(numericalData)


clean_elements_function = function(x){
  x = str_squish(x)
  x = gsub("[{}]", "", x)
  x = gsub("'", '', x)
}

count_function = function(x){
  value = length(unlist(strsplit(as.character(x), ",")))
}

last_element_list_function = function(x){
  value = unlist(strsplit(as.character(x), ","))
  value = as.character(value[length(value)])
  clean_elements_function(value)
}

count_characters_function = function(x){
  x = unlist(x)
  x = nchar(str_squish(x), type = "chars")
}


market_data_preparation = function(market){
  
  market$time = as.Date(market$time)
  
  volume_to_mean = market['volume'] / mean(market$volume)
  volume_to_mean = as.data.frame(volume_to_mean)
  colnames(volume_to_mean) = c("volume_to_mean")
  market = cbind(market,(volume_to_mean))
  
  returnsOpenPrevRaw1_to_volume = market['returnsOpenPrevRaw1'] / market['volume']
  returnsOpenPrevRaw1_to_volume = as.data.frame(returnsOpenPrevRaw1_to_volume)
  colnames(returnsOpenPrevRaw1_to_volume) = c("returnsOpenPrevRaw1_to_volume")
  market = cbind(market,(returnsOpenPrevRaw1_to_volume))
  
  close_to_open = market['close'] / market['open']
  close_to_open = as.data.frame(close_to_open)
  colnames(close_to_open) = c("close_to_open")
  market = cbind(market,(close_to_open))
  
  return((market))
}

news_data_preparation = function(news){
  news['sentence_word_count'] =  news['wordCount'] / news['sentenceCount']
  news['time']= hour(news$time)
  news['sourceTimestamp']= hour(news$sourceTimestamp)
  news['firstCreated']= as.Date(news$firstCreated)
  
  asset.codes.list = convertRowsToList(news['assetCodes'])
  asset.codes.len  = t(as.data.frame(lapply(asset.codes.list, count_function)))
  colnames(asset.codes.len) = c("assetCodesLen")
  news = cbind(news,asset.codes.len)
  news['assetCodes'] = t(as.data.frame(lapply(asset.codes.list, last_element_list_function)))
  asset.codes.list.new = convertRowsToList(news['assetCodes'])
  news['assetCodesLen'] = t(as.data.frame(lapply(asset.codes.list.new, count_characters_function)))
  
  headlines.list = convertRowsToList(news['headline'])
  headlines.len = t(as.data.frame(lapply(headlines.list, count_characters_function)))
  colnames(headlines.len) = c("headlinesLen")
  news = cbind(news,headlines.len)
  
  asset.sentiment.count.dataframe = news%>%
    group_by(assetName,sentimentClass)%>%
    select(time)%>%
    tally
  new.column = asset.sentiment.count.dataframe[match(news$assetName,asset.sentiment.count.dataframe$assetName),3]
  asset.sentiment.count = as.data.frame(new.column)
  colnames(asset.sentiment.count) = c("assetSentimentCount")
  news = cbind(news,asset.sentiment.count)
  
  df_old = as.data.frame(news['headlineTag'])
  a = convertRowsToList(unique(df_old['headlineTag']))
  b = unlist(a)
  x = as.array(clean_elements_function(b))
  y = seq(0,length(x)-1)
  df = as.data.frame(cbind(x,y))
  new.column = df[match(df_old$headlineTag, df$x),2]
  headline.Tag.T = as.data.frame(new.column)
  colnames(headline.Tag.T) = c("headlineTagT")
  news = cbind(news,headline.Tag.T)
  
  colnames(news)[which(names(news) == "assetName")] = "assetNames"
  colnames(news)[which(names(news) == "time")] = "time.news"
  return(news)
}

combined_data_preparation = function(market,news){
  combined_df = merge(x=market,y=news,by.x = c("time", "assetName"), by.y =c("firstCreated", "assetNames"))
  combined_df = na.omit(combined_df)
  return(combined_df)
}

