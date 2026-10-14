function diag_inner(rateTag, m, alpha)
  addpath(genpath('official_code'));
  pkg load optim statistics;
  load(['official_code/Data/proteinFold_Per', rateTag, '.mat']);
  gt = truelabel{1};
  [X1, ind] = findindex(data, index);
  numview = length(X1);
  for iv = 1:numview
      Z{iv} = client_pretrain(X1{iv}, m, alpha, ind(:,iv));
  end
  Zall = server_pretrain(Z, ind, m);
  % --- replicate client_train for view 1 only, printing each inner step ---
  X = X1{1}; Zv = Z{1}; lambda = alpha;
  indexM = find(ind(:,1));
  PZall = Zall;   % P = I
  for it = 1:8
      C = Zv*Zv'; D = X*Zv';
      if cond(C)>1e12, A = D*pinv(C); else, A = D*inv(C); end
      H = A'*A + lambda*eye(m);
      C1 = X'*A + lambda*PZall';
      opt = optimset('Algorithm','interior-point-convex','Display','off');
      for ii = 1:length(indexM)
          ti = -C1(indexM(ii),:);
          Zv(:,indexM(ii)) = quadprog(H, ti', [],[], ones(1,m), 1, zeros(m,1), ones(m,1), [], opt);
      end
      o = norm(X-A*Zv,'fro')^2 + lambda*norm(PZall.*repmat(ind(:,1)',m,1)-Zv,'fro')^2;
      printf('INNER it=%d obj=%.6f fit=%.6f znorm=%.6f\n', it, o, norm(X-A*Zv,'fro')^2, sum(Zv(:).^2));
  end
end
