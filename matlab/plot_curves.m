function plot_curves()
root=fileparts(fileparts(mfilename('fullpath')));
set(0,'DefaultFigureVisible','off');
T=readtable(fullfile(root,'outputs','summary','curves.csv'),'TextType','string');
methods={'BaS-TCP','Greedy-TCP','GA-TCP','MO-TCP','STRaP','SPECTRE*','ART','Cluster-TCP','RL-static*','LEAST-static*','Random Order','Original Order'};
names={'BaS-TCP','Greedy-TCP','GA-TCP','MO-TCP','STRaP','SPECTRE*','ART','Cluster-TCP','RL-static*','LEAST-static*','Random','Original Order'};
colors=[1 0 1;.85 0 0;0 0 1;0 .7 .7;0 .55 0;1 .5 0;.5 0 .63;.55 .27 .07;.7 .6 0;.85 .33 .55;0 0 0;.45 .45 .45];
styles={'-','--','-','--','--','-','-','--','-.','-.',':','-'};
counts=[2849,312,12];totals=[93572.555*10,57388.299777,110];
fig=figure('Units','inches','Position',[1 1 7.2 2.5],'Color','w');
layout=tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
layout.OuterPosition=[0 0 1 0.95];
axesHandles=gobjects(3,1);
for d=1:3
 ax=nexttile(layout);axesHandles(d)=ax;hold(ax,'on');box(ax,'on');h=gobjects(12,1);
 for k=1:12
  U=T(T.Dataset==d & T.Method==methods{k},:);assert(~isempty(U));
  h(k)=plot(ax,U.Budget*totals(d),U.MeanFailingFraction*counts(d),'Color',colors(k,:),...
   'LineStyle',styles{k},'LineWidth',1,'Marker','none');
 end
 budget=.3*totals(d);yl=[0 counts(d)*1.06];
 plot(ax,[budget budget],yl,'--','Color',[.35 .35 .35],'LineWidth',.9,'HandleVisibility','off');
 for k=1:12
  U=T(T.Dataset==d & T.Method==methods{k},:);
  v=interp1(U.Budget*totals(d),U.MeanFailingFraction*counts(d),budget,'linear','extrap');
  plot(ax,[0 budget],[v v],'--','Color',colors(k,:),'LineWidth',.7,'HandleVisibility','off');
 end
 xlim(ax,[0 totals(d)]);ylim(ax,yl);grid(ax,'off');ax.FontSize=8;
 if d<3,xl=xlabel(ax,{'Test Execution Cost (Seconds)',sprintf('(%c) Dataset_%d','a'+d-1,d)},'FontSize',8);
 else,xl=xlabel(ax,{'Number of Test Executions',sprintf('(%c) Dataset_%d','a'+d-1,d)},'FontSize',8);end
 if d<3
  exponentLabel=ax.XAxis.SecondaryLabel;exponentLabel.Units='normalized';
  exponentLabel.Position=[1 1.005 0];exponentLabel.HorizontalAlignment='right';exponentLabel.VerticalAlignment='bottom';
 end
 ylabel(ax,'Failing scenarios detected','FontSize',8);
end
drawnow;
legendGeometry=cell(1,3);
for d=1:3
 legendGeometry{d}=compactLegend(axesHandles(d),names,colors,styles);
end
file=fullfile(root,'outputs','APFD_curves.pdf');
set(fig,'PaperUnits','inches','PaperSize',[7.2 2.5],'PaperPosition',[0 0 7.2 2.5]);
print(fig,file,'-dpdf','-painters');fprintf('EXPORTED %s\n',file);close(fig);
end


function meta=compactLegend(ax,names,colors,styles)
fontSize=5;tokenWidth=6;tokenTextGap=1.8;columnGap=2;
padding=1.2;edgeGap=0.6;rowPitch=7.2;
fontName=ax.FontName;widths=zeros(12,1);probes=gobjects(12,1);
for k=1:12
 probes(k)=text(ax,0,0,names{k},'Units','points','FontSize',fontSize,...
  'FontName',fontName,'Interpreter','none','Visible','off');
end
drawnow;
for k=1:12
 extent=get(probes(k),'Extent');widths(k)=extent(3);assert(widths(k)>0);
end
delete(probes);
axisPixels=getpixelposition(ax,true);dpi=get(0,'ScreenPixelsPerInch');
axisPoints=axisPixels(3:4)*72/dpi;
firstWidth=tokenWidth+tokenTextGap+max(widths(1:6));
secondWidth=tokenWidth+tokenTextGap+max(widths(7:12));
boxWidth=2*padding+firstWidth+columnGap+secondWidth;
boxHeight=2*padding+6*rowPitch;
assert(boxWidth+edgeGap<axisPoints(1) && boxHeight+edgeGap<axisPoints(2));
left=1-(boxWidth+edgeGap)/axisPoints(1);bottom=edgeGap/axisPoints(2);
xl=xlim(ax);yl=ylim(ax);dx=diff(xl);dy=diff(yl);
rectangle(ax,'Position',[xl(1)+left*dx,yl(1)+bottom*dy,...
 boxWidth/axisPoints(1)*dx,boxHeight/axisPoints(2)*dy],...
 'FaceColor','w','EdgeColor','k','LineWidth',0.5,'HandleVisibility','off');
for k=1:12
 col=floor((k-1)/6);row=mod(k-1,6);
 xPoint=padding+col*(firstWidth+columnGap);
 yPoint=boxHeight-padding-(row+0.5)*rowPitch;
 xx=left+[xPoint,xPoint+tokenWidth]/axisPoints(1);
 yy=bottom+yPoint/axisPoints(2);
 line(ax,xl(1)+xx*dx,yl(1)+[yy,yy]*dy,'Color',colors(k,:),...
  'LineStyle',styles{k},'LineWidth',1,'HandleVisibility','off');
 text(ax,left+(xPoint+tokenWidth+tokenTextGap)/axisPoints(1),yy,names{k},...
  'Units','normalized','FontSize',fontSize,'FontName',fontName,'Interpreter','none',...
  'HorizontalAlignment','left','VerticalAlignment','middle','Clipping','on','Margin',0.01);
end
meta=struct('font_pt',fontSize,'column_gap_pt',columnGap,'right_gap_pt',edgeGap,...
 'bottom_gap_pt',edgeGap,'internal_padding_pt',padding,'row_pitch_pt',rowPitch,...
 'box_width_pt',boxWidth,'box_height_pt',boxHeight,'axis_size_pt',axisPoints,...
 'label_widths_pt',widths,'methods',{names});
end
