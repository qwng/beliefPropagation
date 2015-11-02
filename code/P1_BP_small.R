######################################################################################
# Ryan Murphy, Qi Wang, Linna Henry October 2015, Machine Learning
# Homework 2, Problem 1, Small Beleif Propegation
#
#  Do belief propegation on the small burglary network
#
#
######################################################################################

#require(bnstruct)
require(bnlearn)
setwd("C:\\a-My_files_and_folders\\1_Purdue\\_Courses\\D_2015_Fall\\Machine_Learning\\Homework_02\\Problem_01")

# --------------------------------------------
#
# HELPER FUNCTIONS
#
# --------------------------------------------

getScope <- function(vName, CPT){
  cptVarIndx <- which(names(CPT) == vName)
  df.tmp <- as.data.frame(CPT[[cptVarIndx]]$prob)
  dfVarIndx <- which(names(df.tmp) == vName)
  levels(df.tmp[,dfVarIndx])
}

instantiateNodes <- function(iter, nNodes, nodeNames){
  littleList <- vector("list", length = nNodes)
  for(jj in 1:nNodes){
    node <- nodeNames[jj]
    if(node %in% obsVars){
      obsIndx <- grep(node, obsVars)
      observedValue <- obsValues[obsIndx]
      littleList[[jj]] <- GraphNode(node, observedValue)
    }else{
      littleList[[jj]] <- GraphNode(node)
    }
  }
  names(littleList) <- nodeNames
  bpList[[iter]] <<- littleList
}

# ---------------------------------------------------
#  SETUP
#
#  Initialize lists to hold the lambda and pi messages
#    for every node
#
# ---------------------------------------------------

# Set parameters
obsVars <- c("JOHNCALLS") #Name of node we have evidence on
obsValues <- c("TRUE")
names(obsValues) <- obsVars
MAX.ITER <- 100
EPSILON <- 1e-6

#Read in
origCpts <- bnlearn::read.bif("P1_Infile.bif")

# Get names of nodes
nodes <- names(origCpts)
numNodes <- length(origCpts)

#Set variable ordering
remaining <- setdiff(nodes, obsVars)
varOrder <- c(obsVars, remaining[sample(1:length(remaining))])

#Redefine default structure of cpt's
cpts <- vector("list", length = numNodes)
names(cpts) <- nodes

#Define a list to access cpt's, parents, and children
for(ii in 1:length(cpts)){
  cpts[[ii]]$node <- nodes[[ii]]
  cpts[[ii]]$parents <- origCpts[[ii]]$parents
  cpts[[ii]]$children <- origCpts[[ii]]$children
  cpts[[ii]]$probdf <- as.data.frame(origCpts[[ii]]$prob)
  
  if(length(cpts[[ii]]$parents) == 0){
    names(cpts[[ii]]$probdf ) <- c(nodes[[ii]], "Freq")
  }
}


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create class to hold all nodes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GraphNode <- function(node, observedValue = NULL){#observedValue is a character string saying which element was observed, ie "TRUE"
  
  scop <- getScope(node, cpts)
  scopeLength <- length(scop)
  
  ####MAKE laMsg:  Make a list for messages to parents
  pars <- cpts[[node]]$parents
  
  #Test if there are parents
  if(length(pars) == 0){
    hasParents <- FALSE
    laMsg <- NULL
  }else{ #if parents, make a list of messages to each parent
    hasParents <- TRUE
    laMsg <- vector("list", length = length(pars))
    names(laMsg) <- pars
    for(ii in 1:length(pars)){
      parScope <- getScope(pars[ii], cpts)
      laMsg[[ii]] <- vector(length = length(parScope))
      names(laMsg[[ii]]) <- parScope
    }
  }
  #####MAKE lambda value
  laVal <- vector(length = scopeLength)
  names(laVal) <- scop
  
  #####MAKE piMsg:   List for messages to children
  childs	<- cpts[[node]]$children
  if(length(childs) == 0){
    hasChildren <- FALSE
    piMsg <- NULL
  }else{
    hasChildren <- TRUE
    piMsg <- vector("list", length = length(childs))
    names(piMsg) <- childs
    for(ii in 1:length(childs)){
      childScope <- getScope(childs[[ii]], cpts)
      piMsg[[ii]] <- vector(length = length(childScope))
      names(piMsg[[ii]]) <- childScope
    }
  }
  
  #### DEFINE the lambda_X(x), ie the self information
  selfInfo <- vector(length = scopeLength)
  if(is.null(observedValue)){
    selfInfo <- rep(1,scopeLength)
    names(selfInfo) <- scop
  }else{
    selfInfo <- rep(0,scopeLength)
    names(selfInfo) <- scop
    selfInfo[observedValue] <- 1
  }
  
  #####Init class attributes
  me <- list(
    node = node,
    laVal = laVal,
    piVal = laVal,
    bel = laVal,
    laMsg = laMsg,
    piMsg = piMsg,
    parents = pars,
    childs = childs,
    hasParents = hasParents,
    hasChildren = hasChildren,
    numParents = length(pars),
    numChildren = length(childs),
    selfInfo = selfInfo,
    isObserved = (!is.null(observedValue)),
    scopeLength = scopeLength,
    scope = scop
  )
  class(me) <- append(class(me), "GraphNode")
  return(me)
}

#Define member accessors
getNode <- function(graphObj){graphObj$node}
is.GraphNode <- function(graphObj){return(all(class(graphObj) == c("list", "GraphNode")))}

# Instantiate list containing nodes at all iterations objects
bpList <- vector("list", length = MAX.ITER)

# --------------------------------------------------
# Initialization step
# --------------------------------------------------
iter <- 1
instantiateNodes(iter, numNodes, nodes)

#Set all lambda values to 1(vector) except observed
for(ii in 1:numNodes){
  currNode <- bpList[[iter]][[ii]]
  
  # Init all lambdas to one
  nam <- names(bpList[[iter]][[ii]]$laVal)
  bpList[[iter]][[ii]]$laVal <- rep(1, currNode$scopeLength)
  names(bpList[[iter]][[ii]]$laVal) <- nam
  
  # lambda to parents = 1
  if(currNode$hasParents){
    for(kk in 1:currNode$numParents){
      
      ##need to save names to use rep
      nam <- names(bpList[[iter]][[ii]]$laMsg[[kk]])
      bpList[[iter]][[ii]]$laMsg[[kk]] <- rep(1, length(getScope(currNode$parents[kk], cpts)))
      
      names(bpList[[iter]][[ii]]$laMsg[[kk]] ) <- nam
    }
  }else{ #For roots, set pi to be marginal probabilities
    for(kk in 1:currNode$scopeLength){
      tmpVal <- currNode$scope[kk]
      indx <- which(cpts[[currNode$node]]$probdf[,1] == tmpVal)
      bpList[[iter]][[ii]]$piVal[kk] <- cpts[[currNode$node]]$probdf[indx,2]
    }
  }
  # pi to children
  if(currNode$hasChildren){
    for(kk in 1:currNode$numChildren){
      
      nam <- names(bpList[[iter]][[ii]]$piMsg[[kk]])
      bpList[[iter]][[ii]]$piMsg[[kk]] <- rep(1, currNode$scopeLength)
      
      names(bpList[[iter]][[ii]]$piMsg[[kk]]) <- nam
    }
  }
  
}

# ------------------------------------------------
# Iterate until convergence
# ------------------------------------------------

#Init beliefs
for(nn in 1:numNodes){
  nodeName <- nodes[nn]
  sl <- bpList[[1]][[nodeName]]$scopeLength
  bpList[[1]][[nodeName]]$bel <- rep(1/sl, sl)
  names(bpList[[1]][[nodeName]]$bel) <- bpList[[1]][[nodeName]]$scope
}	

checkmarks <- rep(0,numNodes)
names(checkmarks) <- nodes

notConverged <- TRUE
while( notConverged && iter <= MAX.ITER){
  
  iter <- iter + 1
  print(iter)
  flush.console()
  
  bpList[[iter]] <- bpList[[iter-1]]
  
  #----------------------------------------------------------
  #
  #
  ####Use formula to calculate message up (lambda), for each parent
  #
  #
  #----------------------------------------------------------
  #Calculate outgoing messages
  for(ii in 1:numNodes){
    
    # Get next node per variable ordering
    nodeName <- varOrder[ii]
    currNode <- bpList[[iter]][[nodeName]]
    
    # Skip if no parents
    if(currNode$hasParents){
      
      widthCpt <- currNode$numParents + 2 #Num columns in cpt
      
      for(pp in 1:currNode$numParents){
        
        parName <- currNode$parents[pp]
        scopeP <- names(currNode$laMsg[[  parName  ]])
        
        #Loop over every value of u
        for(uu in 1:length(scopeP)){
          
          #Outer sum over X
          sum.x <- 0
          
          for(xx in 1:currNode$scopeLength){
            ###Dragon's clever way to compute product from data frame
            #Copy probdf data frame, without current u
            
            #Get columns in cpt that we need
            uColIndx <- which(names(cpts[[nodeName]]$probdf) == parName)
            #make temp dataframe to hold updated probs
            tmpDf <- cpts[[nodeName]]$probdf[,-uColIndx]
            
            if(currNode$numParents >= 2){
              
              #Modify the probs by including knowledge about parents
              #AKA multiply the PI over k to the cpt probs
              for(dfRow in 1:nrow(tmpDf)){
                for(dfCol in 2:(ncol(tmpDf)-1)){
                  tmpParName <- colnames(tmpDf)[dfCol]
                  u.k <- tmpDf[dfRow,dfCol]
                  tmpDf[dfRow, "Freq"] <- tmpDf[dfRow, "Freq"] * bpList[[iter-1]][[tmpParName]]$piMsg[[nodeName]][u.k]
                }
              }
            }
            
            #Overwrite the probs in cpt with tmpCpt
            condProb <- tmpDf$Freq
            tmpDf <- cpts[[nodeName]]$probdf
            tmpDf$Freq <- condProb
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ###### Now we can sum out x and other parents
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            #sum over cpt pr(x|u)
            x.i <- currNode$scope[xx]
            u.i <- scopeP[uu]
            
            #Get rows corresponding to the realized values we need
            uxRowIndx <- which(cpts[[nodeName]]$probdf[,1] == x.i &
                                 cpts[[nodeName]]$probdf[,uColIndx] == u.i)
            
            #Sum over probability
            lambda.t.x <- bpList[[iter-1]][[nodeName]]$laVal[x.i]
            
            sum.x <- sum.x + ( lambda.t.x ) * sum(tmpDf[ uxRowIndx , c(1, uColIndx, widthCpt)][,"Freq"])
            
          }#Sum over X
          
          #Upade the message with the formula we just poured blood, sweat, and tears into
          bpList[[iter]][[nodeName]]$laMsg[[parName]][u.i] <- sum.x
          
        }# For every value that U can take on
        
        #Normalize!
        alpha <- sum(bpList[[iter]][[nodeName]]$laMsg[[parName]])
        
        if(alpha != 0){
          bpList[[iter]][[nodeName]]$laMsg[[parName]]<- bpList[[iter]][[nodeName]]$laMsg[[parName]] / alpha
        }
      } #For each parent we need to send a msg to
    }#End of calculating outgoing lambda
    
    #----------------------------------------------------------
    #
    #
    ####Use formula to calculate down (pi messages)
    #
    #
    #----------------------------------------------------------
    
    if(currNode$hasChildren){
      
      #Loop over every child of X
      for(cc in 1:currNode$numChildren){
        
        childName <- currNode$childs[cc]
        
        #For every value that x can take on
        for(xx in 1:currNode$scopeLength){
          x.i <- currNode$scope[xx]
          
          prod.k <- 1
          if(currNode$numChildren >= 2){
            
            prodIndx <- setdiff(1:currNode$numChildren, cc)
            
            #Loop over messages from all other children
            for(kk in prodIndx){
              print(kk)
              tmpChildName <- currNode$childs[kk]
              prod.k <- prod.k * bpList[[iter-1]][[tmpChildName]]$laMsg[[nodeName]][x.i]
            }
          }
          bpList[[iter]][[nodeName]]$piMsg[[childName]][x.i] <-  currNode$selfInfo[x.i] * prod.k * bpList[[iter-1]][[nodeName]]$piVal[x.i]
        }#End of loop over values that X takes on
        alpha <- sum(bpList[[iter]][[nodeName]]$piMsg[[childName]])
        if(alpha != 0){
          bpList[[iter]][[nodeName]]$piMsg[[childName]] <- bpList[[iter]][[nodeName]]$piMsg[[childName]] / alpha
        }
        
      }#End of loop over every child of X
      
      
    }####End of calculating the pi message
  }
  #----------------------------------------------------------
  ####Incoming Lambda Message
  #----------------------------------------------------------
  
  #For every value that x can take on
  for(xx in 1:currNode$scopeLength){
    x.i <- currNode$scope[xx]
    if(currNode$isObserved == FALSE){
      #product over messages from below
      prod.j <- 1
      
      if(currNode$numChildren >= 1  && currNode$selfInfo[xx] == 0){
        for(jj in 1:currNode$numChildren){
          childName <- currNode$childs[jj]
          prod.j <- prod.j * bpList[[iter]][[childName]]$laMsg[[nodeName]][x.i]
        }
      }
      bpList[[iter]][[nodeName]]$laVal[x.i] <- prod.j
    }
    
    #----------------------------------------------------------
    ####Incoming Pi Message
    #----------------------------------------------------------
    
    # Skip if no parents
    if(currNode$hasParents){
      
      widthCpt <- currNode$numParents + 2 #Num columns in cpt
      
      
      ###Dragon's clever way to compute product from data frame
      #Copy probdf data frame, without current u
      
      #Get columns in cpt that we need
      tmpDf <- cpts[[nodeName]]$probdf
      
      if(currNode$numParents >= 1){
        
        #Modify the probs by including knowledge about parents
        #AKA multiply the PI over k to the cpt probs
        for(dfRow in 1:nrow(tmpDf)){
          for(dfCol in 2:(ncol(tmpDf)-1)){
            tmpParName <- colnames(tmpDf)[dfCol]
            u.k <- tmpDf[dfRow,dfCol]
            tmpDf[dfRow, "Freq"] <- tmpDf[dfRow, "Freq"] * bpList[[iter]][[tmpParName]]$piMsg[[nodeName]][u.k]
          }
        }
      }
      
      #Overwrite the probs in cpt with tmpCpt
      condProb <- tmpDf$Freq
      tmpDf <- cpts[[nodeName]]$probdf
      tmpDf$Freq <- condProb
      
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ###### Now we can sum out x 
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
      #Get rows corresponding to the realized values of x we need
      xRowIndx <- which(cpts[[nodeName]]$probdf[,1] == x.i )
      
      #Sum over probability
      bpList[[iter]][[nodeName]]$piVal[x.i] <- sum(tmpDf[ xRowIndx , "Freq"])
      
    }
    
    #Normalize!
    alpha <- sum(bpList[[iter]][[nodeName]]$piVal)
    
    if(alpha != 0){
      bpList[[iter]][[nodeName]]$piVal <- bpList[[iter]][[nodeName]]$piVal / alpha
    }
    
    #-----------------------------------------------------------------------
    #Calculate belief
    bpList[[iter]][[nodeName]]$bel <- bpList[[iter]][[nodeName]]$piVal * bpList[[iter]][[nodeName]]$laVal
    eta <- sum(bpList[[iter]][[nodeName]]$bel)
    if(eta > 0){
      bpList[[iter]][[nodeName]]$bel <- bpList[[iter]][[nodeName]]$bel / eta
    }
    
    epsil <- max(abs(  bpList[[iter]][[nodeName]]$bel - bpList[[iter - 1]][[nodeName]]$bel ))
    
    if(epsil <= EPSILON){
      checkmarks[nodeName] <- 1
    }
    
  } #End of calculating lambda value
  
  if(min(checkmarks) == 1){
    notConverged <- FALSE
  }
}
