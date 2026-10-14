function diag_server()
% Isolate server_train: build the same synthetic inputs, run the OFFICIAL
% server_train, and dump inputs + outputs for the Python side to compare.
  addpath(genpath('official_code'));
  pkg load optim statistics;
  rand('twister', 7);
  V = 12; m = 27; n = 694; beta = 0.001;
  Z = cell(V,1); S = cell(V,1);
  ind = double(rand(n,V) > 0.4);
  ind(:,1) = 1;                       % make sure sample 1 exists everywhere
  for iv = 1:V
      Zv = rand(m,n); Zv(:, ind(:,iv)==0) = 0;
      Zv = Zv ./ max(sum(Zv,1), 1e-12);          % simplex columns
      Z{iv} = Zv;
      Sv = rand(m,m); S{iv} = (Sv+Sv')/2;
  end
  Zall = server_pretrain(Z, ind, m);
  P = cell(V,1); for iv=1:V, P{iv} = diag(ones(m,1)); end

  % --- official server_train, instrumented ---
  for iv = 1:V
      S{iv} = (S{iv}+S{iv}')/2;
      D{iv} = diag(sum(S{iv}));
      L{iv} = D{iv} - S{iv};
      L_lmd = eig(L{iv});
      Lmx{iv} = max(L_lmd)*eye(m) - L{iv};
  end
  Zall0 = Zall; P0 = P;
  for iter = 1:6
      ZZ = Zall*Zall';
      for iv = 1:V
          Pv=P{iv}; fobj1 = 0;
          for rep = 1:100
              LPZZ = Lmx{iv}*Pv*ZZ';
              ZZall = Z{iv}*Zall';
              M = 2*beta*LPZZ+ZZall;
              [Um,~,Vm] = svd(M,'econ');
              Pv = Um*Vm';
              fobj2 = fobj1;
              fobj1 = beta*trace(Pv'*LPZZ)+trace(Pv'*ZZall);
              if rep>3 && ((fobj1-fobj2)/fobj1<1e-3), break; end
          end
          P{iv}=Pv;
      end
      Lp = 0; Zp = 0;
      for iv = 1:V
          Lp = Lp+P{iv}'*L{iv}*P{iv};
          Zp = Zp+Z{iv}'*P{iv};
      end
      opt = optimset('Algorithm','interior-point-convex','Display','off');
      for ii = 1:n
          ti = -Zp(ii,:);
          H = sum(ind(ii,:))*eye(m)+2*beta*Lp;
          H = (H+H')/2;
          Zall(:,ii) = quadprog(H,ti',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],opt);
      end
      part1 = 0; part2 = 0;
      for iv = 1:V
          part1 = part1 + norm(P{iv}*Zall.*repmat(ind(:,iv)',m,1)-Z{iv},'fro')^2;
          part2 = part2 + trace(Zall'*P{iv}'*L{iv}*P{iv}*Zall);
      end
      printf('SRV iter=%d part1=%.6f part2=%.6f obj=%.6f\n', iter, part1, part2, part1+2*beta*part2);
  end
  save('-v7','results/_srv_input.mat','Z','S','ind','Zall0','P0','beta','m','n','V');
  printf('SAVED\n');
end
