function diag_server2()
  addpath(genpath('official_code')); pkg load optim statistics;
  rand('twister', 7);
  V = 12; m = 27; n = 694; beta = 0.001;
  ind = double(rand(n,V) > 0.4); ind(:,1) = 1;
  Z = cell(V,1); S = cell(V,1);
  for iv = 1:V
      Zv = rand(m,n); Zv(:, ind(:,iv)==0) = 0; Zv = Zv ./ max(sum(Zv,1),1e-12);
      Z{iv} = Zv; Sv = rand(m,m); S{iv} = (Sv+Sv')/2;
  end
  Zall = server_pretrain(Z, ind, m);
  P = cell(V,1); for iv=1:V, P{iv} = diag(ones(m,1)); end
  for iv = 1:V
      S{iv} = (S{iv}+S{iv}')/2; D{iv} = diag(sum(S{iv})); L{iv} = D{iv}-S{iv};
      Lmx{iv} = max(eig(L{iv}))*eye(m) - L{iv};
  end
  p1 = @(Zal,Pp) sum(cellfun(@(iv) sum(sum((Pp{iv}*Zal.*repmat(ind(:,iv)',m,1)-Z{iv}).^2)), num2cell(1:V)));
  printf('init part1=%.4f\n', p1(Zall,P));
  for iter = 1:3
      ZZ = Zall*Zall';
      for iv = 1:V
          Pv=P{iv}; fobj1=0; nrep=0;
          for rep = 1:100
              nrep = rep;
              LPZZ = Lmx{iv}*Pv*ZZ'; ZZall = Z{iv}*Zall';
              M = 2*beta*LPZZ+ZZall; [Um,~,Vm]=svd(M,'econ'); Pv=Um*Vm';
              fobj2=fobj1; fobj1 = beta*trace(Pv'*LPZZ)+trace(Pv'*ZZall);
              if rep>3 && ((fobj1-fobj2)/fobj1 < 1e-3), break; end
          end
          P{iv}=Pv;
          if iv==1, printf('  iter=%d view1 P-reps=%d\n', iter, nrep); end
      end
      printf('  iter=%d part1_afterP=%.4f\n', iter, p1(Zall,P));
      Lp=0; Zp=0;
      for iv = 1:V, Lp = Lp+P{iv}'*L{iv}*P{iv}; Zp = Zp+Z{iv}'*P{iv}; end
      printf('  iter=%d condH(k=1)=%.2e  min(eig(Lp))=%.4f\n', iter, cond(eye(m)+2*beta*Lp), min(eig(Lp)));
      opt = optimset('Display','off');
      for ii = 1:n
          H = sum(ind(ii,:))*eye(m)+2*beta*Lp; H=(H+H')/2;
          Zall(:,ii) = quadprog(H,-Zp(ii,:)',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],opt);
      end
      printf('  iter=%d part1_afterZ=%.4f\n', iter, p1(Zall,P));
  end
end
