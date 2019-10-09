import matplotlib.pyplot as plt


PAGE_WIDTH = 210

def get_figsize(frac=1, aspect=0.618, width=PAGE_WIDTH):
    """Set aesthetic figure dimensions to avoid scaling in latex.
    Args:
      width: total page width in mm
      fraction: what fraction of the width should the figure take
      ratio: what ratio should the figure be. width / height ratio
      
    Returns
            Dimensions of figure in inches
    """
    width = width * 0.03937 * frac  # convert to inches + scale
    return (width, width * aspect)


def stylize_axes(ax):
    # Drop top-right axis, also 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ticks should point outwards
    ax.xaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(right='off', direction='out', width=1)

    
def paper_config(autolayout=False):
    import matplotlib
    plt.rcdefaults()
    plt.style.use('seaborn-paper')
    
    # Inspired by https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    paper_style = {
        "font.family": "sans-serif",
        "font.sans-serif": 'Arial',
        # Use the same font for mathtext
        "axes.formatter.use_mathtext": True, 
        "mathtext.default": "regular",

        'axes.grid': False,
        'axes.axisbelow': True,  # whether axis gridlines and ticks are below
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        
        
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        # line width
        "lines.linewidth": 0.8,
        
        "xtick.direction": "out",
        "ytick.direction": "out",
        #'xtick.top': False,
        #'ytick.top': False,

        # margins
        # 'axes.xmargin': 0,  # if this is enabled, then there is no margin between the extreme-most point
        # 'axes.ymargin': 0,

        # Colors
        #'axes.color_cycle': ['5DA5DA', 'FAA43A', '60BD68', 'F17CB0', 'B2912F', 'B276B2', 'DECF3F', 'F15854', '4D4D4D'], 
                          # blue,   orange,   green,   pink,     brown,     purple,   yellow,   red,      gray
        "lines.antialiased": True,
        "patch.antialiased": True,

        # Figure export
        "pdf.fonttype": 42,  # save text as text
        "ps.fonttype": 42,
        # Don't cross the borders - careful with this to keep the figure size the same
        "figure.autolayout": autolayout,
        "savefig.dpi": 300,
        "savefig.bbox": 'tight',
        "savefig.pad_inches": 0.05, # Add very small padding to the plot
        "savefig.transparent": True,
        
        "errorbar.capsize": 0.5
    }
    
    # validate that the requested font is indeed installed
    installed_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    if paper_style['font.sans-serif'] not in installed_fonts:
        print(f"WARNING: Font {paper_style['font.sans-serif']} not installed and is required by")

    plt.style.use(paper_style)
