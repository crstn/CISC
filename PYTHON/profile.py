import cProfile, ScriptInProgress, logging

logging.basicConfig(filename='output.log',
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    
cProfile.run('ScriptInProgress.main()')
