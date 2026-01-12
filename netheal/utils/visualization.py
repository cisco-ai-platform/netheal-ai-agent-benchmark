"""
Visualization utilities for network troubleshooting simulation.

This module provides tools for visualizing network graphs, faults, and
diagnostic results to aid in debugging and understanding the environment.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ..network.graph import NetworkGraph, DeviceType
from ..faults.injector import FaultInfo, FaultType


class NetworkVisualizer:
    """Visualizer for network graphs and troubleshooting scenarios."""
    
    def __init__(self):
        """Initialize the network visualizer."""
        self.device_colors = {
            DeviceType.ROUTER: '#FF6B6B',      # Red
            DeviceType.SWITCH: '#4ECDC4',      # Teal
            DeviceType.SERVER: '#45B7D1',      # Blue
            DeviceType.FIREWALL: '#FFA07A',    # Orange
            DeviceType.HOST: '#98D8C8'         # Light Green
        }
        
        self.device_shapes = {
            DeviceType.ROUTER: 's',      # Square
            DeviceType.SWITCH: 'o',      # Circle
            DeviceType.SERVER: '^',      # Triangle
            DeviceType.FIREWALL: 'D',    # Diamond
            DeviceType.HOST: 'h'         # Hexagon
        }
    
    def plot_network(self, network: NetworkGraph, 
                    faults: Optional[List[FaultInfo]] = None,
                    title: str = "Network Topology",
                    figsize: Tuple[int, int] = (12, 8),
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot network topology with optional fault highlighting.
        
        Args:
            network: NetworkGraph to visualize
            faults: List of faults to highlight
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create NetworkX graph for visualization
        G = network.graph.copy()
        
        # Generate layout
        pos = self._generate_layout(G)
        
        # Draw nodes by device type
        for device_type in DeviceType:
            nodes = [n for n, d in G.nodes(data=True) 
                    if d.get('device_type') == device_type]
            
            if nodes:
                # Determine node colors based on status
                node_colors = []
                for node in nodes:
                    if network.is_device_up(node):
                        node_colors.append(self.device_colors[device_type])
                    else:
                        node_colors.append('#CCCCCC')  # Gray for down devices
                
                nx.draw_networkx_nodes(
                    G, pos, nodelist=nodes,
                    node_color=node_colors,
                    node_shape=self.device_shapes[device_type],
                    node_size=800,
                    alpha=0.8,
                    ax=ax
                )
        
        # Draw edges
        edge_colors = []
        edge_styles = []
        for u, v, d in G.edges(data=True):
            if d.get('status') == 'up':
                edge_colors.append('#333333')
                edge_styles.append('-')
            else:
                edge_colors.append('#FF0000')  # Red for down connections
                edge_styles.append('--')
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            style=edge_styles,
            width=2,
            alpha=0.6,
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Highlight faults if provided
        if faults:
            self._highlight_faults(ax, pos, network, faults)
        
        # Add legend
        self._add_legend(ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_diagnostic_results(self, network: NetworkGraph,
                              diagnostic_results: Dict[str, Any],
                              title: str = "Diagnostic Results",
                              figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot network with diagnostic results overlay.
        
        Args:
            network: NetworkGraph being diagnosed
            diagnostic_results: Dictionary of diagnostic tool results
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot network topology
        G = network.graph.copy()
        pos = self._generate_layout(G)
        
        # Draw basic network
        self._draw_basic_network(ax1, G, pos, network)
        ax1.set_title("Network Topology", fontweight='bold')
        
        # Draw diagnostic overlay
        self._draw_diagnostic_overlay(ax2, G, pos, network, diagnostic_results)
        ax2.set_title("Diagnostic Results", fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_episode_summary(self, network: NetworkGraph,
                           ground_truth_fault: FaultInfo,
                           agent_actions: List[Dict[str, Any]],
                           final_diagnosis: Optional[Dict[str, Any]] = None,
                           figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        """
        Plot comprehensive episode summary.
        
        Args:
            network: NetworkGraph for the episode
            ground_truth_fault: The actual fault that was injected
            agent_actions: List of actions taken by the agent
            final_diagnosis: Final diagnosis made by agent
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Network topology with fault
        ax1 = fig.add_subplot(gs[0, 0])
        G = network.graph.copy()
        pos = self._generate_layout(G)
        self._draw_basic_network(ax1, G, pos, network)
        self._highlight_faults(ax1, pos, network, [ground_truth_fault])
        ax1.set_title("Network with Fault", fontweight='bold')
        
        # Action timeline
        ax2 = fig.add_subplot(gs[0, 1:])
        self._plot_action_timeline(ax2, agent_actions)
        
        # Tool usage statistics
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_tool_usage(ax3, agent_actions)
        
        # Success/failure analysis
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_success_analysis(ax4, agent_actions)
        
        # Diagnosis comparison
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_diagnosis_comparison(ax5, ground_truth_fault, final_diagnosis)
        
        plt.suptitle("Episode Summary", fontsize=16, fontweight='bold')
        
        return fig
    
    def _generate_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Generate layout for network visualization."""
        if len(G.nodes()) <= 5:
            return nx.circular_layout(G)
        elif len(G.nodes()) <= 10:
            return nx.spring_layout(G, k=2, iterations=50)
        else:
            return nx.kamada_kawai_layout(G)
    
    def _draw_basic_network(self, ax, G: nx.DiGraph, pos: Dict, network: NetworkGraph):
        """Draw basic network topology."""
        # Draw nodes by type
        for device_type in DeviceType:
            nodes = [n for n, d in G.nodes(data=True) 
                    if d.get('device_type') == device_type]
            
            if nodes:
                node_colors = []
                for node in nodes:
                    if network.is_device_up(node):
                        node_colors.append(self.device_colors[device_type])
                    else:
                        node_colors.append('#CCCCCC')
                
                nx.draw_networkx_nodes(
                    G, pos, nodelist=nodes,
                    node_color=node_colors,
                    node_shape=self.device_shapes[device_type],
                    node_size=600,
                    alpha=0.8,
                    ax=ax
                )
        
        # Draw edges
        edge_colors = ['#333333' if d.get('status') == 'up' else '#FF0000' 
                      for u, v, d in G.edges(data=True)]
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.axis('off')
    
    def _highlight_faults(self, ax, pos: Dict, network: NetworkGraph, faults: List[FaultInfo]):
        """Highlight faults on the network plot."""
        for fault in faults:
            if fault.fault_type == FaultType.DEVICE_FAILURE:
                device_id = fault.details.get('device_id', fault.location)
                if device_id in pos:
                    x, y = pos[device_id]
                    circle = patches.Circle((x, y), 0.15, linewidth=3, 
                                          edgecolor='red', facecolor='none')
                    ax.add_patch(circle)
            
            elif fault.fault_type in [FaultType.LINK_FAILURE, FaultType.MISCONFIGURATION]:
                if '->' in fault.location:
                    source, dest = fault.location.split('->')
                    if source in pos and dest in pos:
                        x1, y1 = pos[source]
                        x2, y2 = pos[dest]
                        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=4, alpha=0.7)
    
    def _draw_diagnostic_overlay(self, ax, G: nx.DiGraph, pos: Dict, 
                               network: NetworkGraph, results: Dict[str, Any]):
        """Draw diagnostic results overlay."""
        self._draw_basic_network(ax, G, pos, network)
        
        # Highlight tested paths
        for key, result in results.items():
            if 'ping' in key or 'traceroute' in key:
                if result.get('success') and 'path' in result:
                    path = result['path']
                    for i in range(len(path) - 1):
                        if path[i] in pos and path[i+1] in pos:
                            x1, y1 = pos[path[i]]
                            x2, y2 = pos[path[i+1]]
                            ax.plot([x1, x2], [y1, y2], 'g-', linewidth=2, alpha=0.8)
    
    def _plot_action_timeline(self, ax, actions: List[Dict[str, Any]]):
        """Plot timeline of agent actions."""
        if not actions:
            ax.text(0.5, 0.5, 'No actions taken', ha='center', va='center')
            ax.set_title("Action Timeline")
            return
        
        action_types = [action.get('type', 'unknown') for action in actions]
        success_status = [action.get('success', False) for action in actions]
        
        colors = ['green' if success else 'red' for success in success_status]
        
        y_pos = range(len(actions))
        ax.barh(y_pos, [1] * len(actions), color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Step {i+1}: {action_types[i]}" for i in range(len(actions))])
        ax.set_xlabel("Actions")
        ax.set_title("Action Timeline")
        ax.invert_yaxis()
    
    def _plot_tool_usage(self, ax, actions: List[Dict[str, Any]]):
        """Plot tool usage statistics."""
        tool_counts = {}
        for action in actions:
            tool = action.get('type', 'unknown')
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        if tool_counts:
            tools = list(tool_counts.keys())
            counts = list(tool_counts.values())
            
            ax.pie(counts, labels=tools, autopct='%1.1f%%', startangle=90)
            ax.set_title("Tool Usage")
        else:
            ax.text(0.5, 0.5, 'No tools used', ha='center', va='center')
            ax.set_title("Tool Usage")
    
    def _plot_success_analysis(self, ax, actions: List[Dict[str, Any]]):
        """Plot success/failure analysis."""
        successful = sum(1 for action in actions if action.get('success', False))
        failed = len(actions) - successful
        
        if successful + failed > 0:
            ax.pie([successful, failed], labels=['Success', 'Failed'], 
                  colors=['green', 'red'], autopct='%1.1f%%')
            ax.set_title("Success Rate")
        else:
            ax.text(0.5, 0.5, 'No actions', ha='center', va='center')
            ax.set_title("Success Rate")
    
    def _plot_diagnosis_comparison(self, ax, ground_truth: FaultInfo, 
                                 diagnosis: Optional[Dict[str, Any]]):
        """Plot comparison between ground truth and agent diagnosis."""
        ax.axis('off')
        
        # Ground truth
        gt_text = f"Ground Truth:\n{ground_truth.fault_type.value}\nat {ground_truth.location}"
        ax.text(0.1, 0.7, gt_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                                          facecolor="lightblue"))
        
        # Agent diagnosis
        if diagnosis:
            diag_text = f"Agent Diagnosis:\n{diagnosis.get('fault_type', 'unknown')}\nat {diagnosis.get('location', 'unknown')}"
            color = "lightgreen" if (diagnosis.get('fault_type') == ground_truth.fault_type and 
                                   diagnosis.get('location') == ground_truth.location) else "lightcoral"
        else:
            diag_text = "Agent Diagnosis:\nNo diagnosis made"
            color = "lightcoral"
        
        ax.text(0.1, 0.3, diag_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                                            facecolor=color))
        
        ax.set_title("Diagnosis Comparison")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _add_legend(self, ax):
        """Add legend for device types."""
        legend_elements = []
        for device_type in DeviceType:
            legend_elements.append(
                plt.Line2D([0], [0], marker=self.device_shapes[device_type], 
                          color='w', markerfacecolor=self.device_colors[device_type],
                          markersize=10, label=device_type.value.title())
            )
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
