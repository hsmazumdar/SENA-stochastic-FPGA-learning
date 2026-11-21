module class9_tree12(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[257] & (~i[353])) | (i[201] & (~i[95]) & i[353]) | ((~i[201]) & (~i[95]) & i[353]) | ((~i[360]) & (~i[257]) & (~i[353])) | ((~i[87]) & i[237] & i[174] & i[95] & i[353]) | (i[161] & (~i[237]) & i[174] & i[95] & i[353]) | ((~i[208]) & (~i[290]) & (~i[174]) & i[95] & i[353]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
