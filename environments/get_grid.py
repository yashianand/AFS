from environments.factored_mdp import FactoredGridWorld
from environments.outdoor_fmdp import OutdoorMDP
from environments.env_to_mdp import EnvToMDP
from ai_safety_gridworlds.environments import side_effects_vase

def readMDPfile(filename, map_name, is_safety_grid=False, is_oracle=None, is_lrtdp=None):
    mdpFile = open(filename, 'r')
    readMDP = eval(mdpFile.read())
    side_effects_vase.GAME_ART = readMDP
    if is_safety_grid==False:
        grid = readMDP[map_name]
        gridWorld = FactoredGridWorld(grid)
    elif is_safety_grid=='outdoor':
        grid = readMDP[map_name]
        gridWorld = OutdoorMDP(grid)
    else:
        grid = side_effects_vase.SideEffectsVaseEnvironment(level=map_name)
        grid.reset()
        gridWorld = EnvToMDP(grid, is_oracle, is_lrtdp)
    return gridWorld
