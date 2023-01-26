% Run state of the art method on real datasets informative
% Methods: GMM, SGMI, SMACD (missing AGML cause code for equal size communities)

function state_of_art_methods_real()
    datasets_list = ["3sources","aucs","BBCSport2view_544","BBC4view_685","WikipediaArticles","cora","citeseer","dkpol","UCI_mfeat"];
    for dataset = datasets_list
        perc=15;
        run(dataset,perc)
    end
end

function run(dataset,perc)
    methods_mean = zeros(3,6); %save mean and std values matrix: row info-1noisy layer-2noisy  layers, column method
    %info
    load("Datasets_Matlab_Real\Matlab_"+dataset+"_knn5_perc"+perc+"_lam1.mat",'A_list','Y_list','labels_list') %load
    sample1 = size(Y_list,2); %sample known labels Y
    K = size(A_list,2); %number layers
    N = size(A_list{1},1); %number nodes
    C = length(unique(labels_list)); %number communities

    %SGMI
    acc_list = SGMI(A_list,Y_list,labels_list,sample1,K,N,C,dataset);
    methods_mean(1,1) = mean(acc_list);
    methods_mean(1,2) = std(acc_list);
    %SMACD
    acc_list = SMACD(A_list,Y_list,labels_list,sample1,K,N,C,dataset);
    methods_mean(1,3) = mean(acc_list);
    methods_mean(1,4) = std(acc_list);
    %GMM
    acc_list = GMM(A_list,Y_list,labels_list,sample1,K,N,C,dataset);
    methods_mean(1,5) = mean(acc_list);
    methods_mean(1,6) = std(acc_list);
    
    
    %noisy
    for k=1:2
        clear A_list Y_list labels_list sample1 K 
        load("Datasets_Matlab_Real\Matlab_"+dataset+"_knn5_perc"+perc+"_lam1_Knoisy"+k+".mat",'A_list','Y_list','labels_list') %load
        K = size(A_list,2); %number layers
        
        sample1 = size(A_list,1); %number sample random matrices
        sample2 = size(Y_list,2); %sample known labels Y

        %SGMI
        acc_list = SGMI_noisy(A_list,Y_list,labels_list,sample1,sample2,K,N,C,dataset,k);
        methods_mean(k+1,1) = mean(acc_list);
        methods_mean(k+1,2) = std(acc_list); %std(mean(acc_list));

        %SMACD
        acc_list = SMACD_noisy(A_list,Y_list,labels_list,sample1,sample2,K,N,C,dataset,k);
        methods_mean(k+1,3) = mean(acc_list);
        methods_mean(k+1,4) = std(acc_list);
        %GMM
        acc_list = GMM_noisy(A_list,Y_list,labels_list,sample1,sample2,K,N,C,dataset,k);
        methods_mean(k+1,5) = mean(acc_list);
        methods_mean(k+1,6) = std(acc_list);
    end 
      
    writematrix(methods_mean,"Results_Real\"+dataset+"\Matlab_acc_"+dataset+"_knn5_perc"+perc+"_lam1.csv")
end


function acc_list = GMM(A_list,Y_list,labels_list,sample1,K,N,C,dataset)
    addpath(genpath('Utils'))
    addpath(genpath('GMM\PM_SSL-master\'))

    %W_cell
    Wcell=cell(1,K);
    for k=1:K
        Wcell{k}=A_list{k};
    end
    
    acc_list = zeros(sample1,1); 
    for r=1:sample1 %for each Y
        %groundTruth
        groundTruth = labels_list';
        if any(groundTruth == 0)
           groundTruth(groundTruth == 0) = C; 
        end
        %y
        y = zeros(N,1);
        [row,~]=find(Y_list{r});
        y(row)=groundTruth(row);
        %p
        p=-1;
        %apply method
        labels = SSL_multilayer_graphs_with_power_mean_laplacian(Wcell, p, y);
        %accuracy
        acc = 1 - get_classification_error(labels, groundTruth, row);
        acc_list(r) = acc; 
    end
    %save
    save("Results_Real\"+dataset+"\GMM\Matlab_"+dataset+"_knn5_lam1.mat", 'acc_list')
end

function acc_list = SGMI(A_list,Y_list,labels_list,sample1,K,N,C,dataset)
    addpath(genpath('Utils'))
    addpath(genpath('SGMI\SMGI\'))
    %L normalized laplacian 
    L=cell(1,K);
    for k=1:K
        W=A_list{k};
        L{k} = GraphLap(W,1);
    end
   

    acc_list = zeros(sample1,1); 
    for r=1:sample1 %for each known labels Y
        %Y
        trY=Y_list{r};
        trY(trY~=0)= 1;
        %apply method
        options.lambda1 = 1;
        options.lambda2 = 1e-3; 
        [F, ~] = SMGI(trY,L,options);
        %groundTruth
        groundTruth = labels_list';
        if any(groundTruth == 0)
           groundTruth(groundTruth == 0) = C; 
        end
        %Communities partition at this iteration
        [~,labels] = max(F,[],2); 
        %known labels
        [row,~]=find(Y_list{r});
        labels(row) = [];
        groundTruth(row) = [];
        %Accuracy
        acc = ((N-length(row))-wrong(groundTruth,labels))/(N-length(row));
        acc_list(r) = acc; 
    end
    %save
    save("Results_Real\"+dataset+"\SGMI\Matlab_"+dataset+"_knn5_lam1.mat", 'acc_list')

end

function acc_list = SMACD(A_list,Y_list,labels_list,sample1,K,N,C,dataset)
    addpath(genpath('Utils'))
    addpath(genpath('SMACD\SMACD-master\SMACD-master\'))

    %W_cell
    Net=cell(1,K);
    for k=1:K
        Net{k}=A_list{k};
    end
    
    acc_list = zeros(sample1,1); 
    for r=1:sample1 %for each Y
        %groundTruth
        groundTruth = labels_list';
        if any(groundTruth == 0)
           groundTruth(groundTruth == 0) = C; 
        end
        %y
        L=Y_list{r};
        L(L~=0)= 1;
        L = full(L);
        [row,~]=find(L);
        %terminology
        R = C; 
        K = size(Net,2);
        [I, J] = size(Net{1});
        X = zeros(I,J,K);
        for i = 1:K
            X(:,:,i) = Net{i};
        end
        X = sptensor(X);
        [labels_i, ~]=SHOCDALL.SHOCD(X,L,R);
        labels=SHOCDALL.permuteLabels(labels_i,groundTruth); % for non-overlapping communities
        labels(row) = [];
        groundTruth(row) = [];
        acc = ((N-length(row))-wrong(groundTruth,labels))/(N-length(row));
        acc_list(r) = acc; 
    end

    %save
    save("Results_Real\"+dataset+"\SMACD\Matlab_"+dataset+"_knn5_lam1.mat", 'acc_list')
end



function acc_list = SGMI_noisy(A_list,Y_list,labels_list,sample1,sample2,K,N,C,dataset,kk)

    addpath(genpath('Utils'))
    addpath(genpath('SGMI\SMGI\'))
    
    acc_list = zeros(sample2,sample1); 
    for rr=1:sample1 %for each random matrices
        %L normalized laplacian 
        L=cell(1,K);
        for k=1:K
            W=A_list{rr,k};
            L{k} = GraphLap(W,1);
        end

        for r=1:sample2 %for each known labels Y
            %Y
            trY=Y_list{rr,r};
            trY(trY~=0)= 1;
            %apply method
            options.lambda1 = 1;
            options.lambda2 = 1e-3;
            [F, ~] = SMGI(trY,L,options);
            %groundTruth
            groundTruth = labels_list';
            if any(groundTruth == 0)
               groundTruth(groundTruth == 0) = C; 
            end
            %Communities partition at this iteration
            [~,labels] = max(F,[],2); 
            %known labels
            [row,~]=find(Y_list{rr,r});
            labels(row) = [];
            groundTruth(row) = [];
            %Accuracy
            acc = ((N-length(row))-wrong(groundTruth,labels))/(N-length(row));
            acc_list(r,rr) = acc; 
        end
    end
    %save
    save("Results_Real\"+dataset+"\SGMI\Matlab_"+dataset+"_knn5_lam1_Knoisy"+kk+".mat", 'acc_list')

end

function acc_list = GMM_noisy(A_list,Y_list,labels_list,sample1,sample2,K,N,C,dataset,kk)
    addpath(genpath('Utils'))
    addpath(genpath('GMM\PM_SSL-master\'))

    acc_list = zeros(sample2,sample1); 
    for rr=1:sample1 %for each random matrices
        %W_cell
        Wcell=cell(1,K);
        for k=1:K
            Wcell{k}=A_list{rr,k};
        end

        for r=1:sample2 %for each Y
            %groundTruth
            groundTruth = labels_list';
            if any(groundTruth == 0)
               groundTruth(groundTruth == 0) = C; 
            end
            %y
            y = zeros(N,1);
            [row,~]=find(Y_list{rr,r});
            y(row)=groundTruth(row);
            %p
            p=-1;
            %apply method
            labels = SSL_multilayer_graphs_with_power_mean_laplacian(Wcell, p, y);
            %accuracy
            acc = 1 - get_classification_error(labels, groundTruth, row);
            acc_list(r,rr) = acc; 
        end
    end
    %save
    save("Results_Real\"+dataset+"\GMM\Matlab_"+dataset+"_knn5_lam1_Knoisy"+kk+".mat", 'acc_list')
end

function acc_list = SMACD_noisy(A_list,Y_list,labels_list,sample1,sample2,K,N,C,dataset,kk)
    addpath(genpath('Utils'))
    addpath(genpath('SMACD\SMACD-master\SMACD-master\'))

    acc_list = zeros(sample2,sample1); 
    for rr=1:sample1 %for each random matrices
        %W_cell
        Net=cell(1,K);
        for k=1:K
            Net{k}=A_list{rr,k};
        end

        for r=1:sample2 %for each Y
            %groundTruth
            groundTruth = labels_list';
            if any(groundTruth == 0)
               groundTruth(groundTruth == 0) = C; 
            end
            %y
            L=Y_list{rr,r};
            L(L~=0)= 1;
            L = full(L);
            [row,~]=find(L);
            %terminology
            R = C; 
            K = size(Net,2);
            [I, J] = size(Net{1});
            X = zeros(I,J,K);
            for i = 1:K
                X(:,:,i) = Net{i};
            end
            X = sptensor(X);
            [labels_i, ~]=SHOCDALL.SHOCD(X,L,R);
            labels=SHOCDALL.permuteLabels(labels_i,groundTruth); % for non-overlapping communities
            labels(row) = [];
            groundTruth(row) = [];
            acc = ((N-length(row))-wrong(groundTruth,labels))/(N-length(row));
            acc_list(r,rr) = acc; 
        end
    end
    %save
    save("Results_Real\"+dataset+"\SMACD\Matlab_"+dataset+"_knn5_lam1_Knoisy"+kk+".mat", 'acc_list')
end




















