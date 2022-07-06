from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch


def cfgArgParser() :
    _parser = default_argument_parser()
    _parser.add_argument("--eval-period", type=int, default=100, help="define evaluation period")
    _parser.add_argument("--epoch", type=int, default=1000, help="number of epoch")
    _parser.add_argument("--batch", type=int, default=16, help="batch size")
    # _parser.add_argument("--class-num", type=int, default=80, help="number of class list")

    _parser.add_argument("--eval-enable", action="store_true", help="enable test")
    _parser.add_argument("--dataset-name",default="mydataset",help="set dataset name")
    _parser.add_argument("--dataset-root",default="dataset",help="set dataset root")
    _parser.add_argument("--image-root",default="image",help="set image root")
    _parser.add_argument("--base-config-file",help="set base config file")
    # _parser.add_argument("--output-path",default="output",help="set dataset name")

    return _parser


if __name__ == '__main__' :
    args = cfgArgParser().parse_args()
    print("Command Line Args:", args)