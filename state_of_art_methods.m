% Run state of the art method on synthetic datasets 
% Methods: GMM, SGMI, AGML, SMACD

function state_of_art_methods()
    name_list = ["info2","noisy","compl"];
    for name = name_list
        if name=="info2"
            cluster_std_list=[5,6,7,8]; 
        else
            cluster_std_list=[2,3,4,5];
        end
        run(name,cluster_std_list)
    end
end

function run(name,cluster_std_list)

    methods_mean = zeros(4,8); %save mean and std values matrix: row cluster_std, column method
    for j = 1:length(cluster_std_list)
        cluster_std = cluster_std_list(j); 
        %load 
        load("Datasets_Matlab\Matlab_theta_N1200_C3_case "+name+"_std"+cluster_std+"_knn5_perc20_lam1_sample5.mat",'A_list','Y_list','labels_list')
        sample1 = size(A_list,1); %number sample random matrices
        K = size(A_list,2); %number layers
        N = size(A_list{1},1); %number nodes
        C = length(unique(labels_list)); %number communities 
    
        %GMM
        acc_list = GMM(cluster_std,A_list,Y_list,labels_list,sample1,K,N,C,name);
        methods_mean(j,1) = mean(acc_list);
        methods_mean(j,2) = std(acc_list);
        %SGMI
        acc_list = SGMI(cluster_std,A_list,Y_list,labels_list,sample1,K,N,C,name);
        methods_mean(j,3) = mean(acc_list);
        methods_mean(j,4) = std(acc_list);
        %AGML
        acc_list = AGML(cluster_std,A_list,Y_list,sample1,K,N,C,name);
        methods_mean(j,5) = mean(acc_list);
        methods_mean(j,6) = std(acc_list);
        %SMACD
        acc_list = SMACD(cluster_std,A_list,Y_list,labels_list,sample1,K,N,C,name);
        methods_mean(j,7) = mean(acc_list);
        methods_mean(j,8) = std(acc_list);

    end
    writematrix(methods_mean,"Results\"+name+"\Matlab_acc_N1200_C3_case "+name+"_std_knn5_perc20_lam1_sample5.csv")
end

function acc_list = GMM(cluster_std,A_list,Y_list,labels_list,sample1,K,N,C,name)
    addpath(genpath('Utils'))
    addpath(genpath('GMM\PM_SSL-master\'))
    
    acc_list = zeros(sample1,1); 
    for r=1:sample1 %for each matrix
        %W_cell
        Wcell=cell(1,K);
        for k=1:K
            Wcell{k}=A_list{r,k};
        end
        %groundTruth
        groundTruth = labels_list(r,:)';
        %groundTruth(groundTruth == 0) = 3; 
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
    save("Results\"+name+"\GMM\Matlab_theta_N1200_C3_case "+name+"_std"+cluster_std+"_knn5_perc20_lam1_sample5.mat", 'acc_list')
end

function acc_list = SGMI(cluster_std,A_list,Y_list,labels_list,sample1,K,N,C,name)
    addpath(genpath('Utils'))
    addpath(genpath('SGMI\SMGI\'))

    acc_list = zeros(sample1,1); 
    for r=1:sample1 %for each matrix
        %L normalized laplacian 
        L=cell(1,K);
        for k=1:K
            W=A_list{r,k};
            L{k} = GraphLap(W,1);
        end
        %Y
        trY=Y_list{r};
        trY(trY~=0)= 1;
        %apply method
        options.lambda1 = 1;
        options.lambda2 = 1e-3;
        [F, ~] = SMGI(trY,L,options);
        %groundTruth
        groundTruth = labels_list(r,:)';
        %groundTruth(groundTruth == 0) = 3; 
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
    save("Results\"+name+"\SGMI\Matlab_theta_N1200_C3_case "+name+"_std"+cluster_std+"_knn5_perc20_lam1_sample5.mat", 'acc_list')

end

function acc_list = AGML(cluster_std,A_list,Y_list,sample1,K,N,C,name)
    addpath(genpath('Utils'))
    addpath(genpath('AGML\AMGL-IJCAI16-master\AMGL-IJCAI16-master\AMGL_Semi\'))

    view_num = K; %number layers
    class_num = C; %number communities 
    each_class_num = N/class_num;
    thresh = 10^-8;
    acc_list = zeros(sample1,1); 
    for r=1:sample1 %for each matrix
        %W_cell
        X=cell(1,view_num );
        for k=1:view_num 
            X{k}=A_list{r,k};
        end        
        %y
        %y = zeros(N,1);
        [row,~]=find(Y_list{r});
        % Each class have the same size of data
        List = row; 
        labeled_N = length(List);
        %part = labeled_N/class_num; 
        List_ = setdiff(1:1:N,List); % the No. of unlabeled data
        samp_label = zeros(N,class_num); % column vector
        for c = 1:class_num
            samp_label((c-1)*each_class_num+(1:each_class_num),c) = ones(each_class_num,1);
        end
        groundtruth = zeros(N,class_num);
        groundtruth(1:labeled_N,:) = samp_label(List,:);
        groundtruth((labeled_N+1):N,:) = samp_label(List_,:);
        F_l = groundtruth(1:labeled_N,:);
        % Construct the affinity matrix for each view data
        for v = 1:view_num
            temp = X{1,v};
            [row_num,col_num] = size(temp);
            fea_v = zeros(row_num,col_num);
            fea_v(:,1:labeled_N) = temp(:,List);
            fea_v(:,(labeled_N+1):N) = temp(:,List_); 
        
            W = constructW_PKN(fea_v); % fea_v is a d_i by n matrix
            d = sum(W);       
            D = diag(d);
            temp_ = diag(sqrt( diag(D).^(-1) ));
            L(1,v) = { eye(N)-temp_*W*temp_ };
        end 
        % Iterately solve the target problem
        maxIter = 100;
        alpha = 1/view_num*ones(1,view_num);
        for iter = 1:maxIter
            % Given alpha, update F_u
            L_sum = zeros(N);
            for v = 1:view_num
                L_sum = L_sum+alpha(v)*L{1,v};
            end
            L_ul = L_sum((labeled_N+1):N,1:labeled_N);
            L_uu = L_sum((labeled_N+1):N,(labeled_N+1):N);
            F_u = -0.5*inv(L_uu)*L_ul*F_l;
            % Given F_u, update alpha
            F = [F_l;F_u];
            for v = 1:view_num
                  alpha(v) = 0.5/sqrt(trace(F'*L{1,v}*F));
            end
            % Calculate objective value
            obj = 0;
            for v = 1:view_num
                  obj = obj+sqrt(trace(F'*L{1,v}*F));
            end
            Obj(iter) = obj;
            if iter>2
                Obj_diff = ( Obj(iter-1)-Obj(iter) )/Obj(iter-1);
                if Obj_diff < thresh
                    break;
                end
            end
        end
        cnt = 0;
        for u = (labeled_N+1):N
            pos = find(F(u,:) == max(F(u,:)));
            y = zeros(1,class_num);
            y(1,pos) = 1;
            if y == groundtruth(u,:)
               cnt = cnt+1;
            end
        end  
        result = cnt/(N-labeled_N);
        acc_list(r) = result;
    end
    %save
    save("Results\"+name+"\AGML\Matlab_theta_N1200_C3_case "+name+"_std"+cluster_std+"_knn5_perc20_lam1_sample5.mat", 'acc_list')
end

function acc_list = SMACD(cluster_std,A_list,Y_list,labels_list,sample1,K,N,C,name)
    addpath(genpath('Utils'))
    addpath(genpath('SMACD\SMACD-master\SMACD-master\'))

    acc_list = zeros(sample1,1); 
    for r=1:sample1 %for each matrix
        %W_cell
        Net=cell(1,K);
        for k=1:K
            Net{k}=A_list{r,k};
        end
        %groundTruth
        groundTruth = labels_list(r,:)';
        %groundTruth(groundTruth == 0) = 3; 
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
    save("Results\"+name+"\SMACD\Matlab_theta_N1200_C3_case "+name+"_std"+cluster_std+"_knn5_perc20_lam1_sample5.mat", 'acc_list')
end


