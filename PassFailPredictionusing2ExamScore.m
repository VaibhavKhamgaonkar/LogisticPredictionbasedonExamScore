% Clear data 
clear ; close all; clc;

%Load Data; make sure the file is present in working directory
data =  load('ex2data1.txt');
X = data(:,1:2);
y = data (:, 3);
m = length(y);

% creating plot function
  function[] = PlottingData(X,y)
    
    % filtering all the posiitve examples' indexes 
      pos = find(y == 1);
    %filtering all the negative examples' indexes
      neg = find(y == 0);
    % ==> visialising the positive and negative data
      %Positive Visiualisation
      plot(X(pos,1), X(pos,2),'markerSize', 7, 'k+', 'linewidth',2);  
      hold on;
     % Negative data Visiualisation
     
      plot(X(neg,1), X(neg,2),'markerSize', 7, 'ko', 'MarkerFaceColor', 'g');  
      hold off;
      xlabel('Exam score 1');
      ylabel ('exam score 2');
      legend( 'selected', 'rejected');
  end  

% Since it is a logstic regression type so we will first seperate the data based on score and selection
PlottingData(X,y);

  
%hypothesis Calculation
  function [h] = hypothesis(z)
    h =  1./(1 + exp(-z) );
  
  end

  
 % Calculation of cost 
    function [J, grad] = Compute_Cost(X,y, m, theta, alpha)
      h = hypothesis(X * theta);
      J = (y' * log(h) + (1-y)' * log(1-h))/(-m);
      grad = (X' * (h-y)) * (alpha/m); 
    end
 
 
 % adding x0 feature to trainign data
 X = [ones(m,1), X];
 
 %Initialising theta,alpha
 theta = zeros(size(X,2),1);
 alpha = 0.1;
 
 %calculating cost 
 cost = Compute_Cost(X,y,m, theta, alpha);
 fprintf('the cost is \n %f \n', cost);
 
 
 % calculating theta using advanced optimisation
% setting up parameter
options =optimset('GradObj', 'on', 'MaxIter', 400);
%calculating theta 

[theta, cost, ExitFlag] = fminunc(@(t)Compute_Cost(X, y, m, t, alpha), theta, options);

% plotting the decison Boundry
  function[] = PlotBoundry(X,y, theta)
   
    PlottingData(X(:, 2:end),y);
    hold on;
    %calculating co ordinates
      plot_x = [min(X(:,2))-2, max(X(:,2))+2];
      plot_y =  (-1./theta(3)).*(theta(2).*plot_x + theta(1));
      %%%(-1./theta(3)).*(theta(2).*plot_x + theta(1));
      plot(plot_x,plot_y);
      legend('Admitted', 'Not admitted', 'Decision Boundary')
      
  end
%decision boundry
  PlotBoundry(X,y,theta); 
   
%Prediction calculation
 prob = hypothesis([1 45 85] * theta);
 fprintf('the predicted probability is \n %f \n', prob);
 